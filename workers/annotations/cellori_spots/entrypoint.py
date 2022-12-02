import argparse
import base64
import json
import sys

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers

import imageio
import numpy as np
from cellori import CelloriSpots

def preview(datasetId, apiUrl, token, params, bimage):
    # Setup helper classes with url and credentials
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)
    threshold = float(workerInterface['Threshold']['value'])

    # Get the tile
    frame = datasetClient.coordinatesToFrameIndex(tile['XY'], tile['Z'], tile['Time'], channel)
    image = datasetClient.getRegion(datasetId, frame=frame).squeeze()

    model = CelloriSpots(model='spots')
    y, _, _ = model._predict(image, stack=False, scale=1)
    counts = y[3].astype(np.uint8)
    thresholded_counts = 255 * (counts > threshold).astype(np.uint8)

    # Convert image to RGB
    rgba = np.ones((*counts.shape, 4), np.uint8) * thresholded_counts[:, :, None]

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
        'Mode': {
            'type': 'select',
            'items': ['Current Z', 'Z-Stack'],
            'default': 'Current Z'
        },
        'Scale': {
            'type': 'number',
            'min': 0,
            'max': 5,
            'default': 1
        },
        'Threshold': {
            'type': 'number',
            'min': 0,
            'max': 9,
            'default': 2.5
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
    """

    # Check whether we need to preview, send the interface, or compute
    request = params.get('request', 'compute')
    if request == 'interface':
        return interface(params['image'], apiUrl, token)
    if request == 'preview':
        return preview(datasetId, apiUrl, token, params, params['image'])

    # roughly validate params
    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    if not all(key in params for key in keys):
        print ("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)

    # Get the Gaussian sigma and threshold from interface values
    stack = workerInterface['Mode']['value'] == 'Z-Stack'
    scale = float(workerInterface['Scale']['value'])
    threshold = float(workerInterface['Threshold']['value'])

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    model = CelloriSpots(model='spots')

    annotationsIds = []

    if stack:

        frames = []

        for z in range(datasetClient.tiles['IndexRange']['IndexZ']):
            frame = datasetClient.coordinatesToFrameIndex(tile['XY'], z, tile['Time'], channel)
            frames.append(datasetClient.getRegion(datasetId, frame=frame).squeeze())

        image = np.stack(frames)

        thresholdCoordinates, _ = model.predict(image, stack=stack, scale=scale, threshold=threshold)

        # Upload annotations TODO: handle connectTo. could be done server-side via special api flag ?
        print("Uploading {} annotations".format(len(thresholdCoordinates)))
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
            annotationsIds.append(annotationClient.createAnnotation(annotation)['_id'])
            print("uploading annotation ", z, x, y)

    else:

        # TODO: will need to iterate or stitch and handle roi and proper intensities
        frame = datasetClient.coordinatesToFrameIndex(tile['XY'], tile['Z'], tile['Time'], channel)
        image = datasetClient.getRegion(datasetId, frame=frame).squeeze()

        thresholdCoordinates, _ = model.predict(image, stack=stack, scale=scale, threshold=threshold)

        # Upload annotations TODO: handle connectTo. could be done server-side via special api flag ?
        print("Uploading {} annotations".format(len(thresholdCoordinates)))
        for [y, x] in thresholdCoordinates:
            annotation = {
                "tags": tags,
                "shape": "point",
                "channel": channel,
                "location": {
                    "XY": assignment['XY'],
                    "Z": assignment['Z'],
                    "Time": assignment['Time']
                },
                "datasetId": datasetId,
                "coordinates": [{"x": float(x), "y": float(y), "z": 0}]
            }
            annotationsIds.append(annotationClient.createAnnotation(annotation)['_id'])

    annotationClient.connectToNearest(connectTo, annotationsIds)

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
