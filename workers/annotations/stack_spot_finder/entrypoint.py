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


def preview(datasetId, apiUrl, token, params, image):
    # Setup helper classes with url and credentials
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)
    thresholdValue = workerInterface['Threshold']['value']
    sigma = workerInterface['Gaussian Sigma']['value']

    # Get the tile
    pngBuffer = datasetClient.getRawImage(tile['XY'], tile['Z'], tile['Time'], channel)
    pngImage = imageio.imread(pngBuffer)

    (width, height) = np.shape(pngImage)

    gaussian = filters.gaussian(pngImage, sigma=sigma, mode='nearest')
    laplacian = filters.laplace(gaussian)
    laplacian = laplacian / np.max(laplacian)

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
    client.setWorkerImagePreview(image, preview)


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Gaussian Sigma': {
            'type': 'number',
            'min': 0,
            'max': 5,
            'default': 2
        },
        'Threshold': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.25
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def main(datasetId, apiUrl, token, params):
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
        request: 'compute', 'preview', 'interface'
        interface: dictionary containing parameters associated to their ids
    """

    # Check whether we need to preview, send the interface, or compute
    request = params.get('request', 'compute')
    if request == 'preview':
        return preview(datasetId, apiUrl, token, params, params['image'])
    if request == 'interface':
        return interface(params['image'], apiUrl, token)

    # roughly validate params
    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    if not all(key in params for key in keys):
        print ("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)

    # Get the Gaussian sigma and threshold from interface values
    threshold = workerInterface['Threshold']['value']
    sigma = workerInterface['Gaussian Sigma']['value']

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    # TODO: will need to iterate or stitch and handle roi and proper intensities
    stack = list()

    for z in range(datasetClient.tiles['IndexRange']['IndexZ']):
        pngBuffer = datasetClient.getRawImage(tile['XY'], z, tile['Time'], channel)
        stack.append(imageio.imread(pngBuffer))

    stack = np.stack(stack)

    # Filter
    gaussian = filters.gaussian(stack, sigma=sigma, mode='nearest')
    laplacian = filters.laplace(gaussian)
    laplacian = laplacian / np.max(laplacian)

    # Find local maxima
    coordinates = peak_local_max(laplacian, min_distance=3)
    values = laplacian[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]

    # Threshold maxima
    index = values > threshold
    thresholdCoordinates = coordinates[index, :]

    # Upload annotations TODO: handle connectTo. could be done server-side via special api flag ?
    print("Uploading {} annotations".format(len(thresholdCoordinates)))
    count = 0
    for [z, y, x] in thresholdCoordinates:
        annotation = {
            "tags": tags,
            "shape": "point",
            "channel": channel,
            "location": {
                "XY": assignment['XY'],
                "Z": int(z),
                "Time": assignment['Time']
            },
            "datasetId": datasetId,
            "coordinates": [{"x": float(x), "y": float(y), "z": 0}]
        }
        annotationClient.createAnnotation(annotation)
        print("uploading annotation ", z, x, y)
        if count > 1000:  # TODO: arbitrary limit to avoid flooding the server if threshold is too big
            break
        count = count + 1


if __name__ == '__main__':

    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in a circle around point annotations')

    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str,
                        required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    main(args.datasetId, args.apiUrl, args.token, json.loads(args.parameters))
