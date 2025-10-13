import base64
import argparse
import json
import sys
import random
import timeit
import time
import threading
from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendProgress  # , sendError


import imageio
import numpy as np

from skimage import filters
from skimage.feature import peak_local_max

from shapely.geometry import Polygon

# --- signal debug shim ---
import signal, threading, os

_SIGNAL_SEEN = threading.Event()
_LAST_SIG = {"num": None}

def _log_signal(signum):
    try:
        name = signal.Signals(signum).name
    except Exception:
        name = str(signum)
    print(json.dumps({
        "type": "debug",
        "event": "signal_received",
        "signal": name,
        "signum": int(signum),
        "pid": os.getpid()
    }))
    sys.stdout.flush()

def _on_signal(signum, frame):
    # only log the first time, but remember which one
    if not _SIGNAL_SEEN.is_set():
        _LAST_SIG["num"] = int(signum)
        _SIGNAL_SEEN.set()
        _log_signal(signum)

# Catch what Docker/`docker stop`/tini will forward
for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT):
    try:
        signal.signal(_s, _on_signal)
    except Exception:
        pass
# --- end signal debug shim ---


# REMOVE THE BELOW
def preview(datasetId, apiUrl, token, params, bimage):
    # Setup helper classes with url and credentials
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    keys = ["assignment", "channel", "connectTo",
            "tags", "tile", "workerInterface"]
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(
        *keys)(params)
    thresholdValue = float(workerInterface['Threshold'])
    sigma = float(workerInterface['Gaussian Sigma'])

    # Get the tile
    frame = datasetClient.coordinatesToFrameIndex(
        tile['XY'], tile['Z'], tile['Time'], channel)
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
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Square size': {
            'type': 'number',
            'min': 0,
            'max': 30,
            'default': 10,
            'tooltip': 'The size of the square annotations to generate.',
            'unit': 'Pixels'
        },
        'Number of random annotations': {
            'type': 'number',
            'min': 0,
            'max': 300000,
            'default': 100,
            'unit': 'annotations',
            'vueAttr': {
                'title': 'Number of random annotations'
            }
        },
        'Batch XY': {
            'type': 'text',
            'required': True,
            'tooltip': 'hello tooltip'
        },
        'Batch Z': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the frames you want to connect',
                'persistentPlaceholder': True,
                'filled': True,
                'title': 'Frames to connect'
            }
        },
        'Batch Time': {
            'type': 'text'
        },
        'Channel test': {
            'type': 'channel',
            'vueAttr': {
                'title': 'test Channel information'
            },
            'tooltip': 'This is a test tooltip for the channel field. Pick the right channel for your data.\n A very very very very very very looooooooong tooltip here. What do you think about it?'
            # 'required': True
        }
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def sendHeartbeat():
    """Sends a heartbeat message to keep the connection alive."""
    print(json.dumps({"type": "heartbeat"}))
    sys.stdout.flush()


def heartbeat_loop(interval=10):
    while True:
        time.sleep(interval)
        sendHeartbeat()


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
    keys = ["assignment", "channel", "connectTo",
            "tags", "tile", "workerInterface"]
    if not all(key in params for key in keys):
        print("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(
        *keys)(params)

    annotationSize = float(workerInterface['Square size'])
    annotationNumber = float(workerInterface['Number of random annotations'])
    # batch_xy = workerInterface.get('Batch XY', None)
    # batch_z = workerInterface.get('Batch Z', None)
    # batch_time = workerInterface.get('Batch Time', None)

    # batch_xy = batch_argument_parser.process_range_list(batch_xy)
    # batch_z = batch_argument_parser.process_range_list(batch_z)
    # batch_time = batch_argument_parser.process_range_list(batch_time)

    # if batch_xy is None:
    #     batch_xy = [tile['XY'] + 1]
    # if batch_z is None:
    #     batch_z = [tile['Z'] + 1]
    # if batch_time is None:
    #     batch_time = [tile['Time'] + 1]

    # Get the Gaussian sigma and threshold from interface values
    # annulus_size = float(workerInterface['Annulus size'])

    heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    heartbeat_thread.start()

    sendProgress(0.1, "Starting worker", "Starting")

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    tile_width = datasetClient.tiles['tileWidth']
    tile_height = datasetClient.tiles['tileHeight']

    workerClient = workers.UPennContrastWorkerClient(
        datasetId, apiUrl, token, params)
    # annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)

    # Provided snippets
    annotationSize = float(workerInterface['Square size'])
    annotationNumber = float(workerInterface['Number of random annotations'])

    tile_width = datasetClient.tiles['tileWidth']
    tile_height = datasetClient.tiles['tileHeight']

    # Create a list to hold the generated annotations
    theAnnotations = []

    sendProgress(0.2, "Starting random square generation",
                 "Generating annotations")

    # Generate random annotations
    for i in range(int(annotationNumber)):
        # Generate a random center point for the square, ensuring it won't go off the edge of the field
        x = random.uniform(annotationSize/2, tile_width - annotationSize/2)
        y = random.uniform(annotationSize/2, tile_height - annotationSize/2)

        # Generate the four corners of the square
        square_coords = [(x - annotationSize/2, y - annotationSize/2),
                         (x + annotationSize/2, y - annotationSize/2),
                         (x + annotationSize/2, y + annotationSize/2),
                         (x - annotationSize/2, y + annotationSize/2)]

        # Define the new annotation
        new_annotation = {
            "tags": tags,  # *** NEED TO UPDATE TO ADD A NEW TAG ****
            "shape": "polygon",
            "channel": channel,
            "location": {
                "XY": tile['XY'],
                "Z": tile['Z'],
                "Time": tile['Time']
            },
            "datasetId": datasetId,
            "coordinates": [{"x": float(coord[0]), "y": float(coord[1])} for coord in square_coords]
        }

        # Append the new annotation to the list
        theAnnotations.append(new_annotation)
        fraction_done = (i + 1) / annotationNumber
        sendProgress(fraction_done, "Generating random squares",
                     f"Generated {i + 1} of {int(annotationNumber)} annotations")

    # sendError("test")

    # print(json.dumps({"error": "test", "title": "testError", "info": "testMessage"}))
    print(json.dumps({"error": "test", "title": "testError",
          "info": "This is just a test error message. May in future include some information on how to resolve it.", "type": "error"}))
    sys.stdout.flush()

    # Below is a test without adding the "info" field
    print(json.dumps(
        {"error": "test", "title": "testError2", "type": "error"}))
    sys.stdout.flush()

    time.sleep(3)

    print(json.dumps(
        {"warning": "test", "title": "testWarning", "type": "warning"}))
    sys.stdout.flush()

    print(json.dumps({"warning": "test", "title": "testWarning",
          "info": "This is just a test warning message. May in future include some information on how to resolve it.", "type": "warning"}))
    sys.stdout.flush()

    start_time = timeit.default_timer()
    # Send the annotations to the server
    # for annotation in theAnnotations:
    #     annotationClient.createAnnotation(annotation)
    annotationClient.createMultipleAnnotations(theAnnotations)
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Executed the code in: {execution_time} seconds")

    # TODO: will need to iterate or stitch and handle roi and proper intensities
    # frame = datasetClient.coordinatesToFrameIndex(tile['XY'], tile['Z'], tile['Time'], channel)
    # image = datasetClient.getRegion(datasetId, frame=frame).squeeze()


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in a circle around point annotations')

    parser.add_argument('--datasetId', type=str,
                        required=False, action='store')
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
