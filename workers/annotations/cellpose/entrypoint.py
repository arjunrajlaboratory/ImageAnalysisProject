import argparse
import json
import sys

from functools import partial
from itertools import product

import annotation_client.workers as workers

import numpy as np  # library for array manipulation
import deeptile
from deeptile.extensions.segmentation import cellpose_segmentation
from deeptile.extensions.stitch import stitch_polygons

from shapely.geometry import Polygon

from worker_client import WorkerClient


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Model': {
            'type': 'select',
            'items': ['cyto', 'cyto2', 'cyto3', 'nuclei'],
            'default': 'cyto3',
            'displayOrder': 3
        },
        'Nuclei Channel': {
            'type': 'channel',
            'default': -1,
            'required': False,
            'displayOrder': 4
        },
        'Cytoplasm Channel': {
            'type': 'channel',
            'default': -1,
            'required': False,
            'displayOrder': 5
        },
        'Diameter': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10,
            'displayOrder': 6
        },
        'Tile Size': {
            'type': 'number',
            'min': 0,
            'max': 2048,
            'default': 1024,
            'displayOrder': 9
        },
        'Tile Overlap': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.1,
            'displayOrder': 10
        },
        'Batch XY': {
            'type': 'text',
            'displayOrder': 0
        },
        'Batch Z': {
            'type': 'text',
            'displayOrder': 1
        },
        'Batch Time': {
            'type': 'text',
            'displayOrder': 2
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'displayOrder': 8,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 10,
            'default': 0,
            'displayOrder': 7,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def run_model(image, cellpose, tile_size, tile_overlap, padding, smoothing):

    dt = deeptile.load(image)
    image = dt.get_tiles(tile_size=(tile_size, tile_size), overlap=(tile_overlap, tile_overlap))

    polygons = cellpose(image)
    polygons = stitch_polygons(polygons)

    if padding != 0:
        dilated_polygons = []
        for polygon in polygons:
            polygon = Polygon(polygon)
            dilated_polygon = polygon.buffer(padding)
            dilated_polygons.append(list(dilated_polygon.exterior.coords))
    else:
        dilated_polygons = polygons

    if smoothing > 0:
        smoothed_polygons = []
        for polygon in dilated_polygons:
            smoothed_polygon = Polygon(polygon).simplify(smoothing)
            smoothed_polygons.append(list(smoothed_polygon.exterior.coords))
        return smoothed_polygons
    else:
        return dilated_polygons


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

    # Get the model and diameter from interface values
    model = worker.workerInterface['Model']
    nuclei_channel = worker.workerInterface.get('Nuclei Channel', None)
    cytoplasm_channel = worker.workerInterface.get('Cytoplasm Channel', None)
    diameter = float(worker.workerInterface['Diameter'])
    tile_size = int(worker.workerInterface['Tile Size'])
    tile_overlap = float(worker.workerInterface['Tile Overlap'])
    padding = float(worker.workerInterface['Padding'])
    smoothing = float(worker.workerInterface['Smoothing'])

    stack_channels = []
    if model in ['cyto', 'cyto2', 'cyto3']:
        if (cytoplasm_channel is not None) and (cytoplasm_channel > -1):
            stack_channels.append(cytoplasm_channel)
    if (nuclei_channel is not None) and (nuclei_channel > -1):
        stack_channels.append(nuclei_channel)
    if len(stack_channels) == 2:
        channels = (0, 1)
    elif len(stack_channels) == 1:
        channels = (0, 0)
    else:
        raise ValueError("No cytoplasmic or nuclei channels selected.")

    cellpose = cellpose_segmentation(model_parameters={'gpu': True, 'model_type': model}, eval_parameters={'diameter': diameter, 'channels': channels}, output_format='polygons')
    f_process = partial(run_model, cellpose=cellpose, tile_size=tile_size, tile_overlap=tile_overlap, padding=padding, smoothing=smoothing)

    worker.process(f_process, f_annotation='polygon', stack_channels=stack_channels, progress_text='Running Cellpose')


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
