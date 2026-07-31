import argparse
import json
import sys
from functools import partial

import annotation_client.workers as workers
from annotation_client.utils import sendError, sendWarning
import annotation_utilities.annotation_tools as annotation_tools


import girder_utils
from girder_utils import MODELS_DIR

from shapely.geometry import Polygon

from worker_client import WorkerClient, geometry_to_polygon_coords

from models_config import (
    BASE_MODELS, DEFAULT_MODEL, build_cellpose_parameters, build_model_items)


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    girder_models = [model['name']
                     for model in girder_utils.list_girder_models(client.client)[0]]
    # Reserved base labels win over any custom model of the same name (see
    # build_model_items); this keeps the dropdown consistent with compute()'s routing.
    models = build_model_items(girder_models)

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
            'default': DEFAULT_MODEL,
            'tooltip': 'cellpose-sam runs the cpsam_v2 checkpoint (the current default).\n'
                       'Choose "cellpose-sam (legacy cpsam)" to reproduce results from the '
                       'original April 2025 model. Custom trained models are also listed here.',
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
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 10,
            'default': 0.7,
            'tooltip': 'Smoothing is used to simplify the polygons. A value of 0.7 is a good default.',
            'displayOrder': 8,
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'unit': 'pixels',
            'tooltip': 'Padding will expand (or, if negative, subtract) from the polygon. A value of 0 means no padding.',
            'displayOrder': 9,
        },
        'Tile Size': {
            'type': 'number',
            'min': 0,
            'max': 2048,
            'default': 1024,
            'unit': 'pixels',
            'tooltip': 'The worker will split the image into tiles of this size. If they are too large, the Cellpose model may not be able to run on them.',
            'displayOrder': 10,
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
            'displayOrder': 11,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def get_slot_channels(workerInterface):
    """Resolve the selected input-slot channels into an ordered channel list.

    Slot 1 is required, Slots 2 and 3 are optional, and if multiple channels are
    checked in a slot only the first is used (with a warning). The raw values are
    parsed by annotation_tools.get_selected_channels rather than .items(), which
    crashed on the malformed list shape ([0]) found in a saved tool config.
    """
    slots = []
    for slot in ('Channel for Slot 1', 'Channel for Slot 2', 'Channel for Slot 3'):
        try:
            slots.append(annotation_tools.get_selected_channels(
                workerInterface.get(slot), slot))
        except ValueError as exc:
            sendError(f"Could not read the channel selection for {slot}.",
                      info=str(exc))
            raise

    slot1, slot2, slot3 = slots
    stack_channels = []

    if not slot1:
        sendError("No channel selected for Slot 1. This is a required field.")
        raise ValueError("No channel selected for Slot 1.")

    for name, channels in (('Slot 1', slot1), ('Slot 2', slot2), ('Slot 3', slot3)):
        if not channels:
            continue
        if len(channels) > 1:
            sendWarning(
                f"Multiple channels selected for {name} ({channels}). Using the first: {channels[0]}.")
        stack_channels.append(channels[0])

    return stack_channels


def run_model(image, cellpose, tile_size, tile_overlap, padding, smoothing):

    # Lazy import: keeps deeptile off the interface/startup path (~seconds). See todo/worker-startup-latency.md
    import deeptile
    from deeptile.extensions.stitch import stitch_polygons

    dt = deeptile.load(image)
    image = dt.get_tiles(tile_size=(tile_size, tile_size),
                         overlap=(tile_overlap, tile_overlap))

    polygons = cellpose(image)
    polygons = stitch_polygons(polygons)

    if padding != 0:
        dilated_polygons = []
        for polygon in polygons:
            # A negative buffer (shrinking) can erode a small object away to an
            # empty geometry or pinch it into a MultiPolygon, so normalize the
            # result and drop anything that no longer forms a valid polygon.
            dilated_polygon = Polygon(polygon).buffer(padding)
            dilated_polygons.extend(geometry_to_polygon_coords(dilated_polygon))
    else:
        dilated_polygons = polygons

    if smoothing > 0:
        smoothed_polygons = []
        for polygon in dilated_polygons:
            smoothed_polygon = Polygon(polygon).simplify(smoothing)
            smoothed_polygons.extend(geometry_to_polygon_coords(smoothed_polygon))
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

    # Lazy import: keeps deeptile off the interface/startup path (~seconds). See todo/worker-startup-latency.md
    from deeptile.extensions.segmentation import cellpose_segmentation

    worker = WorkerClient(datasetId, apiUrl, token, params)

    # Get the model and post-processing parameters from interface values
    model = worker.workerInterface['Model']
    tile_size = int(worker.workerInterface['Tile Size'])
    tile_overlap = float(worker.workerInterface['Tile Overlap'])
    padding = float(worker.workerInterface['Padding'])
    smoothing = float(worker.workerInterface['Smoothing'])

    stack_channels = get_slot_channels(worker.workerInterface)

    print(f"Using channels for Cellpose-SAM input (slots 1, 2, 3): {stack_channels}")

    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    models_dir = MODELS_DIR
    if model not in BASE_MODELS:
        try:
            downloaded_model = girder_utils.download_girder_model(
                client.client, model)
        except FileNotFoundError as exc:
            sendError("Custom model unavailable.", info=str(exc))
            raise
        models_dir = downloaded_model.parent

    # Print the contents of the models directory
    print(f"Models directory contents: {list(MODELS_DIR.glob('*'))}")

    cellpose_parameters = build_cellpose_parameters(model, models_dir)
    cellpose = cellpose_segmentation(
        **cellpose_parameters, output_format='polygons')
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
