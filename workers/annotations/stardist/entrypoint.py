import argparse
import json
import sys

from functools import partial

import annotation_client.workers as workers

import numpy as np
import deeptile
from stardist.models import StarDist2D
from deeptile.extensions.stitch import stitch_polygons

from shapely.geometry import Polygon

from worker_client import WorkerClient


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Model': {
            'type': 'select',
            'items': ['2D_versatile_fluo', '2D_versatile_he'],
            'default': '2D_versatile_fluo',
            'displayOrder': 3
        },
        'Channel': {
            'type': 'channel',
            'default': -1,
            'required': True,
            'displayOrder': 4
        },
        'Probability Threshold': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.5,
            'displayOrder': 5
        },
        'NMS Threshold': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.4,
            'displayOrder': 6
        },
        'Tile Size': {
            'type': 'number',
            'min': 0,
            'max': 2048,
            'default': 1024,
            'displayOrder': 7
        },
        'Tile Overlap': {
            'type': 'number',
            'min': 0,
            'max': 1,
            'default': 0.1,
            'displayOrder': 8
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
            'displayOrder': 9,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 10,
            'default': 1,
            'displayOrder': 10,
        },
    }
    client.setWorkerImageInterface(image, interface)


def run_model(image, model, prob_thresh, nms_thresh, tile_size, tile_overlap, padding, smoothing):
    dt = deeptile.load(image)
    image = dt.get_tiles(tile_size=(tile_size, tile_size), overlap=(tile_overlap, tile_overlap))

    def stardist_segmentation(tile):
        labels, _ = model.predict_instances(tile, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        return labels

    polygons = deeptile.lift(stardist_segmentation)(image)
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
    worker = WorkerClient(datasetId, apiUrl, token, params)

    model_name = worker.workerInterface['Model']
    channel = worker.workerInterface['Channel']
    prob_thresh = float(worker.workerInterface['Probability Threshold'])
    nms_thresh = float(worker.workerInterface['NMS Threshold'])
    tile_size = int(worker.workerInterface['Tile Size'])
    tile_overlap = float(worker.workerInterface['Tile Overlap'])
    padding = float(worker.workerInterface['Padding'])
    smoothing = float(worker.workerInterface['Smoothing'])

    model = StarDist2D.from_pretrained(model_name)

    f_process = partial(run_model, model=model, prob_thresh=prob_thresh, nms_thresh=nms_thresh,
                        tile_size=tile_size, tile_overlap=tile_overlap, padding=padding, smoothing=smoothing)

    worker.process(f_process, f_annotation='polygon', stack_channels=[channel], progress_text='Running StarDist')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute StarDist segmentation on images')

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