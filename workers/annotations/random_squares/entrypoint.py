import argparse
import json
import sys
import random

import annotation_client.workers as workers
from worker_client import WorkerClient


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    interface = {
        'Random Squares': {
            'type': 'notes',
            'value': 'Generates random square polygon annotations within the image bounds. '
                     'Useful for testing and development.',
            'displayOrder': 0,
        },
        'Square size': {
            'type': 'number',
            'min': 1,
            'max': 200,
            'default': 10,
            'tooltip': 'The side length of each square annotation in pixels.',
            'unit': 'pixels',
            'displayOrder': 1,
        },
        'Number of squares': {
            'type': 'number',
            'min': 1,
            'max': 300000,
            'default': 100,
            'tooltip': 'How many random square annotations to generate per tile position.',
            'unit': 'annotations',
            'displayOrder': 2,
        },
        'Batch XY': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'XY positions to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 3,
        },
        'Batch Z': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Z slices to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 4,
        },
        'Batch Time': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Time points to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 5,
        },
    }
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    worker = WorkerClient(datasetId, apiUrl, token, params)

    square_size = float(worker.workerInterface['Square size'])
    num_squares = int(float(worker.workerInterface['Number of squares']))

    tile_width = worker.datasetClient.tiles['tileWidth']
    tile_height = worker.datasetClient.tiles['tileHeight']

    half = square_size / 2.0

    def generate_squares(image):
        """Generate random square polygons.

        The image argument is required by WorkerClient.process() but is
        unused here because square generation is image-independent.
        """
        polygons = []
        for _ in range(num_squares):
            cx = random.uniform(half, tile_width - half)
            cy = random.uniform(half, tile_height - half)
            square = [
                (cx - half, cy - half),
                (cx + half, cy - half),
                (cx + half, cy + half),
                (cx - half, cy + half),
            ]
            polygons.append(square)
        return polygons

    worker.process(generate_squares, f_annotation='polygon',
                   progress_text='Generating random squares')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate random square polygon annotations')

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
