import base64
import argparse
import json
import sys
import random
import timeit

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers

import imageio
import numpy as np

from skimage import filters
from skimage.feature import peak_local_max

from shapely.geometry import Polygon
import utils


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
        'Number of random point annotations': {
            'type': 'number',
            'min': 0,
            'max': 20000,
            'default': 200
        },
        'Batch XY': {
            'type': 'text'
        },
        'Batch Z': {
            'type': 'text'
        },
        'Batch Time': {
            'type': 'text'
        }
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

    annotationNumber = float(workerInterface['Number of random point annotations'])
    batch_xy = workerInterface.get('Batch XY', None)
    batch_z = workerInterface.get('Batch Z', None)
    batch_time = workerInterface.get('Batch Time', None)

    batch_xy = utils.process_range_list(batch_xy)
    batch_z = utils.process_range_list(batch_z)
    batch_time = utils.process_range_list(batch_time)

    if batch_xy is None:
        batch_xy = [tile['XY'] + 1]
    if batch_z is None:
        batch_z = [tile['Z'] + 1]
    if batch_time is None:
        batch_time = [tile['Time'] + 1]

    # Get the Gaussian sigma and threshold from interface values
    #annulus_size = float(workerInterface['Annulus size'])

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    tile_width = datasetClient.tiles['tileWidth']
    tile_height = datasetClient.tiles['tileHeight']

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)

    tile_width = datasetClient.tiles['tileWidth']
    tile_height = datasetClient.tiles['tileHeight']

    # Create a list to hold the generated annotations
    theAnnotations = []

    # Generate random annotations
    for _ in range(int(annotationNumber)):
        # Generate a random point
        x = random.uniform(0, tile_width)
        y = random.uniform(0, tile_height)
        
        # Define the new annotation
        new_annotation = {
            "tags": tags, 
            "shape": "point",
            "channel": channel,
            "location": {
                        "XY": tile['XY'],
                        "Z": tile['Z'],
                        "Time": tile['Time']
                        },
            "datasetId": datasetId,
            "coordinates": [{"x": float(x), "y": float(y)}]
        }
        
        # Append the new annotation to the list
        theAnnotations.append(new_annotation)

    start_time = timeit.default_timer()
    # Send the annotations to the server
    #for annotation in theAnnotations:
    #    annotationClient.createAnnotation(annotation)
    annotationClient.createMultipleAnnotations(theAnnotations)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Executed the code in: {execution_time} seconds")

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
