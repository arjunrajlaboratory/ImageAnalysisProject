import argparse
import json
import sys

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

from itertools import product
from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers

import numpy as np
from piscis import Piscis
from piscis.paths import MODELS_DIR

import utils


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    models = sorted(path.stem for path in MODELS_DIR.glob('*'))

    # Available types: number, text, tags, layer
    interface = {
        'Model': {
            'type': 'select',
            'items': models,
            'default': models[-1]
        },
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
            'default': 1.0
        },
        'Assign to Nearest Z': {
            'type': 'select',
            'items': ['Yes', 'No'],
            'default': 'Yes'
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

    # Get the Gaussian sigma and threshold from interface values
    model_name = workerInterface['Model']
    stack = workerInterface['Mode'] == 'Z-Stack'
    scale = float(workerInterface['Scale'])
    threshold = float(workerInterface['Threshold'])
    nearest_z = workerInterface['Assign to Nearest Z'] == 'Yes'
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

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    model = Piscis(model_name=model_name, batch_size=2)

    if stack:

        for xy, time in product(batch_xy, batch_time):

            xy -= 1
            time -= 1

            frames = []

            for z in range(datasetClient.tiles['IndexRange']['IndexZ']):
                frame = datasetClient.coordinatesToFrameIndex(xy, z, time, channel)
                frames.append(datasetClient.getRegion(datasetId, frame=frame).squeeze())

            image = np.stack(frames)

            thresholdCoordinates = model.predict(image, stack=stack, scale=scale, threshold=threshold, intermediates=False)
            thresholdCoordinates[:, -2:] += 0.5

            # Upload annotations TODO: handle connectTo. could be done server-side via special api flag ?
            print("Uploading {} annotations".format(len(thresholdCoordinates)))
            annotation_list = []
            for [z, y, x] in thresholdCoordinates:
                annotation = {
                    "tags": tags,
                    "shape": "point",
                    "channel": channel,
                    "location": {
                        "XY": xy,
                        "Z": int(z) if nearest_z else assignment['Z'],
                        "Time": time
                    },
                    "datasetId": datasetId,
                    "coordinates": [{"x": float(x), "y": float(y), "z": 0}]
                }
                annotation_list.append(annotation)
            annotationsIds = [a['_id'] for a in annotationClient.createMultipleAnnotations(annotation_list)]
            if len(connectTo['tags']) > 0:
                annotationClient.connectToNearest(connectTo, annotationsIds)

    else:

        for xy, z, time in product(batch_xy, batch_z, batch_time):

            xy -= 1
            z -= 1
            time -= 1

            # TODO: will need to iterate or stitch and handle roi and proper intensities
            frame = datasetClient.coordinatesToFrameIndex(xy, z, time, channel)
            image = datasetClient.getRegion(datasetId, frame=frame).squeeze()

            thresholdCoordinates = model.predict(image, stack=stack, scale=scale, threshold=threshold, intermediates=False)
            thresholdCoordinates += 0.5

            # Upload annotations TODO: handle connectTo. could be done server-side via special api flag ?
            print("Uploading {} annotations".format(len(thresholdCoordinates)))
            annotation_list = []
            for [y, x] in thresholdCoordinates:
                annotation = {
                    "tags": tags,
                    "shape": "point",
                    "channel": channel,
                    "location": {
                        "XY": xy,
                        "Z": z,
                        "Time": time
                    },
                    "datasetId": datasetId,
                    "coordinates": [{"x": float(x), "y": float(y), "z": 0}]
                }
                annotation_list.append(annotation)
            annotationsIds = [a['_id'] for a in annotationClient.createMultipleAnnotations(annotation_list)]
            if len(connectTo['tags']) > 0:
                annotationClient.connectToNearest(connectTo, annotationsIds)

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
