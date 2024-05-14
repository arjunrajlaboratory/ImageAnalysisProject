import base64
import argparse
import json
import sys

from operator import itemgetter

import annotation_client.tiles as tiles
import annotation_client.workers as workers

import imageio
import numpy as np

from worker_client import WorkerClient

from functools import partial
from skimage import feature, filters, measure


def preview(datasetId, apiUrl, token, params, bimage):
    # Setup helper classes with url and credentials
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)
    threshold = float(workerInterface['Threshold'])
    sigma = float(workerInterface['Gaussian Sigma'])

    # Get the tile
    frame = datasetClient.coordinatesToFrameIndex(tile['XY'], tile['Z'], tile['Time'], channel)
    image = datasetClient.getRegion(datasetId, frame=frame).squeeze()

    (width, height) = np.shape(image)

    log = filters.laplace(filters.gaussian(image, sigma=sigma))

    # Compute the threshold indexes
    indices = log > threshold

    # Convert image to RGB
    rgba = np.zeros((width, height, 4), np.uint8)

    # Paint threshold areas red
    rgba[indices] = [255, 255, 255, 255]

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
        'Sigma': {
            'type': 'number',
            'min': 0,
            'max': 5,
            'default': 2
        },
        'Threshold': {
            'type': 'text',
            'default': 0.001
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


def find_spots(image, stack, sigma, threshold):

    log = filters.laplace(filters.gaussian(image, sigma=sigma))

    if stack:
        labels = measure.label(log > threshold)
        coords = np.array([region['centroid'] for region in measure.regionprops(labels)], dtype=int)

    else:
        coords = feature.peak_local_max(log, min_distance=1, threshold_abs=threshold, exclude_border=False)

    return coords


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

    worker = WorkerClient(datasetId, apiUrl, token, params)

   # Get the Gaussian sigma and threshold from interface values
    stack = worker.workerInterface['Mode'] == 'Z-Stack'
    threshold = float(worker.workerInterface['Threshold'])
    sigma = float(worker.workerInterface['Gaussian Sigma'])

    f_process = partial(find_spots, stack=stack, sigma=sigma, threshold=threshold)
    worker.process(f_process, f_annotation='point', stack_zs='all' if stack else None, progress_text='Running Spot Finder')


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
