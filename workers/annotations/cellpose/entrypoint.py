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
        'Cellpose': {
            'type': 'notes',
            'value': 'This tool runs the Cellpose model to segment the image into cells.',
            'displayOrder': 0,
        },
        'Batch XY': {
            'type': 'text',
            'vueAttrs': {
               'placeholder': 'ex. 1-3, 5-8',
               'label': 'Enter the XY positions you want to iterate over',
               'persistentPlaceholder': True,
               'filled': True,
            },
            'displayOrder': 1
        },
        'Batch Z': {
            'type': 'text',
            'vueAttrs': {
               'placeholder': 'ex. 1-3, 5-8',
               'label': 'Enter the Z slices you want to iterate over',
               'persistentPlaceholder': True,
               'filled': True,
            },
            'displayOrder': 2
        },
        'Batch Time': {
            'type': 'text',
            'vueAttrs': {
               'placeholder': 'ex. 1-3, 5-8',
               'label': 'Enter the Time points you want to iterate over',
               'persistentPlaceholder': True,
               'filled': True,
            },
            'displayOrder': 3
        },
        'Model': {
            'type': 'select',
            'items': ['cyto', 'cyto2', 'cyto3', 'nuclei'],
            'default': 'cyto3',
            'tooltip': 'cyto3 is the most accurate for cells, whereas nuclei is best for finding nuclei.\n'
                       'You will need to select a nuclei and cytoplasm channel in both cases.\n'
                       'If you select nuclei, put the nucleus channel in both the Nuclei Channel and Cytoplasm Channel fields.',
            'displayOrder': 5
        },
        'Nuclei Channel': {
            'type': 'channel',
            # 'default': -1,  # -1 means no channel
            'required': False,
            'displayOrder': 6
        },
        'Cytoplasm Channel': {
            'type': 'channel',
            # 'default': -1,  # -1 means no channel
            'required': False,
            'displayOrder': 7
        },
        'Diameter': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10,
            'tooltip': 'The diameter of the cells in the image. Choose as close as you can\n'
                       'because the model is most accurate when the diameter is close to the actual cell diameter.',
            'displayOrder': 8
        },
        'Padding and Smoothing': {
            'type': 'notes',
            'value': 'Padding will expand (or, if negative, subtract) from the polygon. Smoothing is used to simplify the polygons.',
            'displayOrder': 9,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 10,
            'default': 0.3,
            'tooltip': 'Smoothing is used to simplify the polygons. A value of 0.3 is a good default.',
            'displayOrder': 10,
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'tooltip': 'Padding will expand (or, if negative, subtract) from the polygon. A value of 0 means no padding.',
            'displayOrder': 11,
        },
        'Tiling': {
            'type': 'notes',
            'value': 'Tiling is used to speed up processing by breaking the image into smaller tiles. '
                     'Make sure that the largest object is smaller than the overlap; i.e., if your tile size is 1024 and overlap is 0.1, '
                     'then the largest object should be less than 102 pixels in its longest dimension.',
            'displayOrder': 12,
        },
        'Tile Size': {
            'type': 'number',
            'min': 0,
            'max': 2048,
            'default': 1024,
            'tooltip': 'The worker will split the image into tiles of this size. If they are too large, the Cellpose model may not be able to run on them.',
            'displayOrder': 13
        },
        'Tile Overlap': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.1,
            'tooltip': 'The amount of overlap between tiles. A value of 0.1 means that the tiles will overlap by 10%, which is 102 pixels if the tile size is 1024.\n'
                       'Make sure your objects are smaller than the overlap; i.e., if your tile size is 1024 and overlap is 0.1, '
                       'then the largest object should be less than 102 pixels in its longest dimension.',
            'displayOrder': 14
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
