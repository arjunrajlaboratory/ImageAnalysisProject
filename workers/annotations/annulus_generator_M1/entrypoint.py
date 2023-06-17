import base64
import argparse
import json
import sys

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers

import imageio
import numpy as np

from skimage import filters
from skimage.feature import peak_local_max

from shapely.geometry import Polygon


# REMOVE THE BELOW
def preview(datasetId, apiUrl, token, params, bimage):
    # Setup helper classes with url and credentials
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)
    thresholdValue = float(workerInterface['Threshold'])
    sigma = float(workerInterface['Gaussian Sigma'])

    # Get the tile
    frame = datasetClient.coordinatesToFrameIndex(tile['XY'], tile['Z'], tile['Time'], channel)
    image = datasetClient.getRegion(datasetId, frame=frame).squeeze()

    (width, height) = np.shape(image)

    gaussian = filters.gaussian(image, sigma=sigma, mode='nearest')
    laplacian = filters.laplace(gaussian)

    # Compute the threshold indexes
    index = laplacian > thresholdValue

    # Convert image to RGB
    rgba = np.zeros((width, height, 4), np.uint8)

    # Paint threshold areas red
    rgba[index] = [255, 0, 0, 255]

    # Generate an output data-uri from the threshold image
    outputPng = imageio.imwrite('<bytes>', rgba, format='png')
    data64 = base64.b64encode(outputPng)
    dataUri = 'data:image/png;base64,' + data64.decode('ascii')

    # Send the preview object to the server
    preview = {
        'image': dataUri
    }
    client.setWorkerImagePreview(bimage, preview)


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Annulus size': {
            'type': 'number',
            'min': 0,
            'max': 30,
            'default': 10
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    """
    params (could change):
        configurationId,
        datasetId,
        description: tool description,
        type: tool type,
        id: tool id,
        name: tool name,
        image: docker image,
        channel: annotation channel,
        assignment: annotation assignment ({XY, Z, Time}),
        tags: annotation tags (list of strings),
        tile: tile position (TODO: roi) ({XY, Z, Time}),
        connectTo: how new annotations should be connected
    """

    # roughly validate params
    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    if not all(key in params for key in keys):
        print ("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)

    # Get the Gaussian sigma and threshold from interface values
    annulus_size = float(workerInterface['Annulus size'])

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)


    # We need at least one annotation
    if len(annotationList) == 0:
        return

    for annotation in annotationList:
        #polygon = np.array([list(coordinate.values())[1::-1] for coordinate in annotation['coordinates']], dtype=int)
        # In the below line, you may want to say list(coordinate.values())[0:2] to avoid any potential Z coordinate that could be stored in there as well.
        # polygon = np.array([list(coordinate.values()) for coordinate in annotation['coordinates']], dtype=int)
        polygon = np.array([(coordinate['x'], coordinate['y']) for coordinate in annotation['coordinates']], dtype=int)
        # Create a shapely Polygon object
        poly = Polygon(polygon)

        # Buffer (dilate) the polygon
        dilated_poly = poly.buffer(annulus_size)

        outer_coords = list(dilated_poly.exterior.coords)
        inner_coords = list(poly.exterior.coords)
        annulus_coords = outer_coords + inner_coords[::-1]
        #annulus = Polygon(annulus_coords)

        # Convert dilated polygon back to list of coordinates
        #dilated_polygon_coords = np.array(list(dilated_poly.exterior.coords), dtype=int)
        #dilated_polygon_coords = np.array(list(annulus.exterior.coords), dtype=int)
        annulus_coords = np.array(list(annulus_coords), dtype=int)

        myTags = annotation['tags'].copy()
        myTags.append('annulus')
        #print(myTags)

        # Create a new annotation
        #annotation['tags'].append('annulus') # Append "annulus" as a tag.
        new_annotation = {
            "tags": myTags, # *** NEED TO UPDATE TO ADD A NEW TAG ****
            "shape": "polygon",
            "channel": annotation['channel'],
            "location": annotation['location'],
            "datasetId": datasetId,
            # "coordinates": [{"x": float(x), "y": float(y), "z": 0} for x, y in annulus_coords] #original line. Not sure why we need to specify Z.
            "coordinates": [{"x": float(x), "y": float(y)} for x, y in annulus_coords]
        }

        # Upload the new annotation
        annotationClient.createAnnotation(new_annotation)

    # TODO: will need to iterate or stitch and handle roi and proper intensities
    #frame = datasetClient.coordinatesToFrameIndex(tile['XY'], tile['Z'], tile['Time'], channel)
    #image = datasetClient.getRegion(datasetId, frame=frame).squeeze()




if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in a circle around point annotations')

    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str,
                        required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    params = json.loads(args.parameters)
    datasetId = args.datasetId
    apiUrl = args.apiUrl
    token = args.token

    match args.request:
        case 'compute':
            compute(datasetId, apiUrl, token, params)
        case 'interface':
            interface(params['image'], apiUrl, token)
        case 'preview':
            preview(datasetId, apiUrl, token, params, params['image'])
