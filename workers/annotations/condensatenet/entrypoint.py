import argparse
import json
import sys
from functools import partial

import annotation_client.workers as workers

import numpy as np
from shapely.geometry import Polygon

import deeptile
from deeptile.core.lift import lift
from deeptile.core.data import Output
from deeptile.core.utils import compute_dask
from deeptile.extensions.segmentation import mask_to_polygons
from deeptile.extensions.stitch import stitch_polygons

from worker_client import WorkerClient

from condensatenet import CondensateNetPipeline

from annotation_client.utils import sendProgress


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Batch XY': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the XY positions you want to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 0
        },
        'Batch Z': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the Z slices you want to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 1
        },
        'Batch Time': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the Time points you want to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 2
        },
        'Probability Threshold': {
            'type': 'number',
            'min': 0.0,
            'max': 1.0,
            'default': 0.15,
            'tooltip': 'Minimum confidence for detection (0-1)',
            'displayOrder': 3,
        },
        'Min Size': {
            'type': 'number',
            'min': 1,
            'max': 1000,
            'default': 15,
            'unit': 'pixels',
            'tooltip': 'Minimum condensate size in pixels',
            'displayOrder': 4,
        },
        'Max Size': {
            'type': 'number',
            'min': 10,
            'max': 10000,
            'default': 600,
            'unit': 'pixels',
            'tooltip': 'Maximum condensate size in pixels',
            'displayOrder': 5,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.3,
            'tooltip': 'Polygon simplification tolerance',
            'displayOrder': 6,
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'unit': 'pixels',
            'tooltip': 'Padding will expand (or, if negative, subtract) from the polygon. A value of 0 means no padding.',
            'displayOrder': 7,
        },
        'Tile Size': {
            'type': 'number',
            'min': 256,
            'max': 2048,
            'default': 1024,
            'unit': 'pixels',
            'tooltip': 'The worker will split the image into tiles of this size. If they are too large, the model may run out of memory.',
            'displayOrder': 8,
        },
        'Tile Overlap': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.1,
            'unit': 'Fraction',
            'tooltip': 'The amount of overlap between tiles. A value of 0.1 means tiles overlap by 10%. '
                       'Make sure your objects are smaller than the overlap region.',
            'displayOrder': 9,
        },
    }
    client.setWorkerImageInterface(image, interface)


def condensatenet_segmentation(prob_threshold, min_size, max_size):
    """Generate lifted function for CondensateNet segmentation.

    Parameters
    ----------
    prob_threshold : float
        Minimum probability for detection.
    min_size : int
        Minimum condensate size in pixels.
    max_size : int
        Maximum condensate size in pixels.

    Returns
    -------
    func_segment : Callable
        Lifted function for the CondensateNet segmentation algorithm.
    """

    # Load CondensateNet model once
    sendProgress(0, "Loading model", "Initializing CondensateNet...")
    pipeline = CondensateNetPipeline.from_local(
        "/models/condensatenet",
        prob_threshold=prob_threshold,
        min_size=min_size,
        max_size=max_size
    )

    @lift
    def _func_segment(tile, index, tile_index, stitch_index, tiling):
        tile = compute_dask(tile)

        # Squeeze out channel dimension if present (from stack_channels)
        tile = np.squeeze(tile)

        # Pad tile to dimensions divisible by 32 for FPN compatibility
        original_shape = tile.shape[:2]
        pad_h = (32 - original_shape[0] % 32) % 32
        pad_w = (32 - original_shape[1] % 32) % 32
        if pad_h > 0 or pad_w > 0:
            tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='reflect')

        # Run segmentation
        mask = pipeline.segment(tile)

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            mask = mask[:original_shape[0], :original_shape[1]]

        # Convert mask to polygons using DeepTile's mask_to_polygons
        polygons = mask_to_polygons(mask, index, tile_index, stitch_index, tiling)
        return polygons

    def func_segment(tiles):
        return _func_segment(
            tiles,
            tiles.index_iterator,
            tiles.tile_indices_iterator,
            tiles.stitch_indices_iterator,
            tiles.profile.tiling
        )

    return func_segment


def run_model(image, condensatenet, tile_size, tile_overlap, padding, smoothing):
    """Run CondensateNet with tiling support.

    Parameters
    ----------
    image : ndarray
        Input image.
    condensatenet : Callable
        Lifted CondensateNet segmentation function.
    tile_size : int
        Size of tiles in pixels.
    tile_overlap : float
        Fraction of overlap between tiles.
    padding : float
        Dilation/erosion amount for polygons.
    smoothing : float
        Polygon simplification tolerance.

    Returns
    -------
    list
        List of polygon coordinates.
    """
    dt = deeptile.load(image)
    tiles = dt.get_tiles(
        tile_size=(tile_size, tile_size),
        overlap=(tile_overlap, tile_overlap)
    )

    polygons = condensatenet(tiles)
    polygons = stitch_polygons(polygons)

    # Apply padding (dilation/erosion)
    if padding != 0:
        dilated_polygons = []
        for polygon in polygons:
            polygon = Polygon(polygon)
            dilated_polygon = polygon.buffer(padding)
            if not dilated_polygon.is_empty:
                dilated_polygons.append(list(dilated_polygon.exterior.coords))
        polygons = dilated_polygons

    # Apply smoothing
    if smoothing > 0:
        smoothed_polygons = []
        for polygon in polygons:
            smoothed_polygon = Polygon(polygon).simplify(smoothing, preserve_topology=True)
            if not smoothed_polygon.is_empty and smoothed_polygon.is_valid:
                smoothed_polygons.append(list(smoothed_polygon.exterior.coords))
        return smoothed_polygons
    else:
        return polygons


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
        tile: tile position ({XY, Z, Time}),
        connectTo: how new annotations should be connected
    """

    worker = WorkerClient(datasetId, apiUrl, token, params)

    # Get parameters from interface
    prob_threshold = float(worker.workerInterface['Probability Threshold'])
    min_size = int(worker.workerInterface['Min Size'])
    max_size = int(worker.workerInterface['Max Size'])
    smoothing = float(worker.workerInterface['Smoothing'])
    padding = float(worker.workerInterface['Padding'])
    tile_size = int(worker.workerInterface['Tile Size'])
    tile_overlap = float(worker.workerInterface['Tile Overlap'])

    channel = worker.channel

    # Create the lifted segmentation function (loads model once)
    condensatenet = condensatenet_segmentation(prob_threshold, min_size, max_size)

    # Create the processing function with all parameters bound
    f_process = partial(
        run_model,
        condensatenet=condensatenet,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        padding=padding,
        smoothing=smoothing
    )

    # Process all batches using WorkerClient
    worker.process(
        f_process,
        f_annotation='polygon',
        stack_channels=[channel],
        progress_text='Running CondensateNet'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CondensateNet Segmentation Worker')

    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str, required=True, action='store')

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
