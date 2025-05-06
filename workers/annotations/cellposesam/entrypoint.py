import argparse
import json
import sys
from pathlib import Path
from functools import partial
from itertools import product

import annotation_client.workers as workers
from annotation_client.utils import sendError, sendWarning, sendProgress

import numpy as np  # library for array manipulation
import deeptile
from deeptile.extensions.segmentation import cellpose_segmentation
from deeptile.extensions.stitch import stitch_polygons

import girder_utils
from girder_utils import CELLPOSE_DIR, MODELS_DIR

from shapely.geometry import Polygon

from worker_client import WorkerClient

BASE_MODELS = ['cellpose-sam']


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # models = sorted(path.stem for path in MODELS_DIR.glob('*'))
    models = BASE_MODELS
    girder_models = [model['name']
                     for model in girder_utils.list_girder_models(client.client)[0]]
    models = sorted(list(set(models + girder_models)))

    # Available types: number, text, tags, layer
    interface = {
        'Cellpose-SAM': {
            'type': 'notes',
            'value': 'This tool runs the Cellpose-SAM model to segment the image into cells. '
                     '<a href="https://docs.nimbusimage.com/documentation/analyzing-image-data-with-objects-connections-and-properties/tools-for-making-objects#cellpose-for-automated-cell-finding" target="_blank">Learn more</a>',
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
            'displayOrder': 1,
        },
        'Batch Z': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the Z slices you want to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 2,
        },
        'Batch Time': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the Time points you want to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 3,
        },
        'Model': {
            'type': 'select',
            'items': models,
            'default': 'cellpose-sam',
            'tooltip': 'cellpose-sam is the base model',
            'noCache': True,
            'displayOrder': 4,
        },
        'Channel for Slot 1': {
            'type': 'channelCheckboxes',
            'tooltip': "Select source channel(s) for the model's first input slot. If multiple are selected, only the first will be used. This slot is required.",
            'displayOrder': 5
        },
        'Channel for Slot 2': {
            'type': 'channelCheckboxes',
            'tooltip': "Select source channel(s) for the model's second input slot. If multiple are selected, only the first will be used. (Optional)",
            'displayOrder': 6
        },
        'Channel for Slot 3': {
            'type': 'channelCheckboxes',
            'tooltip': "Select source channel(s) for the model's third input slot. If multiple are selected, only the first will be used. (Optional)",
            'displayOrder': 7
        },
        'Diameter': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10,
            'unit': 'pixels',
            'tooltip': 'The diameter of the cells in the image. Choose as close as you can\n'
                       'because the model is most accurate when the diameter is close to the actual cell diameter.',
            'displayOrder': 8,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 10,
            'default': 0.7,
            'tooltip': 'Smoothing is used to simplify the polygons. A value of 0.7 is a good default.',
            'displayOrder': 9,
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'unit': 'pixels',
            'tooltip': 'Padding will expand (or, if negative, subtract) from the polygon. A value of 0 means no padding.',
            'displayOrder': 10,
        },
        'Tile Size': {
            'type': 'number',
            'min': 0,
            'max': 2048,
            'default': 1024,
            'unit': 'pixels',
            'tooltip': 'The worker will split the image into tiles of this size. If they are too large, the Cellpose model may not be able to run on them.',
            'displayOrder': 11,
        },
        'Tile Overlap': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.1,
            'unit': 'Fraction',
            'tooltip': 'The amount of overlap between tiles. A value of 0.1 means that the tiles will overlap by 10%, which is 102 pixels if the tile size is 1024.\n'
                       'Make sure your objects are smaller than the overlap; i.e., if your tile size is 1024 and overlap is 0.1, '
                       'then the largest object should be less than 102 pixels in its longest dimension.',
            'displayOrder': 12,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def run_model(image, cellpose, tile_size, tile_overlap, padding, smoothing):

    dt = deeptile.load(image)
    image = dt.get_tiles(tile_size=(tile_size, tile_size),
                         overlap=(tile_overlap, tile_overlap))

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
    diameter = float(worker.workerInterface['Diameter'])
    tile_size = int(worker.workerInterface['Tile Size'])
    tile_overlap = float(worker.workerInterface['Tile Overlap'])
    padding = float(worker.workerInterface['Padding'])
    smoothing = float(worker.workerInterface['Smoothing'])

    # Process new channel selections
    slot1_channel_str_keys = [k for k, v in worker.workerInterface.get(
        'Channel for Slot 1', {}).items() if v]
    slot2_channel_str_keys = [k for k, v in worker.workerInterface.get(
        'Channel for Slot 2', {}).items() if v]
    slot3_channel_str_keys = [k for k, v in worker.workerInterface.get(
        'Channel for Slot 3', {}).items() if v]

    stack_channels = []

    if not slot1_channel_str_keys:
        sendError("No channel selected for Slot 1. This is a required field.")
        raise ValueError("No channel selected for Slot 1.")
    if len(slot1_channel_str_keys) > 1:
        sendWarning(
            f"Multiple channels selected for Slot 1 ({slot1_channel_str_keys}). Using the first: {slot1_channel_str_keys[0]}.")
    stack_channels.append(int(slot1_channel_str_keys[0]))

    if slot2_channel_str_keys:
        if len(slot2_channel_str_keys) > 1:
            sendWarning(
                f"Multiple channels selected for Slot 2 ({slot2_channel_str_keys}). Using the first: {slot2_channel_str_keys[0]}.")
        stack_channels.append(int(slot2_channel_str_keys[0]))

    if slot3_channel_str_keys:
        if len(slot3_channel_str_keys) > 1:
            sendWarning(
                f"Multiple channels selected for Slot 3 ({slot3_channel_str_keys}). Using the first: {slot3_channel_str_keys[0]}.")
        stack_channels.append(int(slot3_channel_str_keys[0]))

    if not stack_channels:  # Should technically be caught by slot 1 check, but as a safeguard.
        sendError("No channels were selected for processing.")
        raise ValueError("No channels selected for processing.")

    print(f"Using channels for Cellpose-SAM input (slots 1, 2, 3): {stack_channels}")

    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    if model not in BASE_MODELS:
        girder_utils.download_girder_model(client.client, model)

    # Print the contents of the models directory
    print(f"Models directory contents: {list(MODELS_DIR.glob('*'))}")

    if model in BASE_MODELS:
        cellpose = cellpose_segmentation(
            model_parameters={'gpu': True}, eval_parameters={}, output_format='polygons')
    else:
        # Get the full path to the model
        model_path = str(MODELS_DIR / model)
        cellpose = cellpose_segmentation(model_parameters={'gpu': True, 'pretrained_model': model_path}, eval_parameters={
                                         'diameter': diameter}, output_format='polygons')
    f_process = partial(run_model, cellpose=cellpose, tile_size=tile_size,
                        tile_overlap=tile_overlap, padding=padding, smoothing=smoothing)

    worker.process(f_process, f_annotation='polygon',
                   stack_channels=stack_channels, progress_text='Running Cellpose-SAM')


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

    if args.request == 'compute':
        compute(datasetId, apiUrl, token, params)
    elif args.request == 'interface':
        interface(params['image'], apiUrl, token)
