import argparse
import json
import sys
import time
import random

import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendWarning, sendError
from worker_client import WorkerClient


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    interface = {
        # --- Notes type ---
        'About this worker': {
            'type': 'notes',
            'value': 'This is a <b>sample interface worker</b> that demonstrates all available '
                     'interface types, tooltips, display ordering, vueAttrs, and messaging '
                     '(progress, warnings, errors).',
            'displayOrder': 0,
        },
        # --- Number type ---
        'Sample number': {
            'type': 'number',
            'min': 0,
            'max': 100,
            'default': 42,
            'tooltip': 'A sample number input. Demonstrates min/max/default and unit display.',
            'unit': 'pixels',
            'displayOrder': 1,
        },
        # --- Text type ---
        'Sample text': {
            'type': 'text',
            'default': 'Hello, NimbusImage!',
            'tooltip': 'A sample text input with vueAttrs for placeholder and label.',
            'vueAttrs': {
                'placeholder': 'Enter some text...',
                'label': 'Sample text field',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 2,
        },
        # --- Select type ---
        'Sample select': {
            'type': 'select',
            'items': ['Option A', 'Option B', 'Option C'],
            'default': 'Option A',
            'tooltip': 'A sample dropdown select input.',
            'displayOrder': 3,
        },
        # --- Checkbox type ---
        'Sample checkbox': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'A sample checkbox. When checked, the worker will send an extra warning message.',
            'displayOrder': 4,
        },
        # --- Channel type ---
        'Sample channel': {
            'type': 'channel',
            'tooltip': 'A sample single-channel selector.',
            'displayOrder': 5,
        },
        # --- Channel checkboxes type ---
        'Sample channel checkboxes': {
            'type': 'channelCheckboxes',
            'tooltip': 'A sample multi-channel checkbox selector.',
            'displayOrder': 6,
        },
        # --- Tags type ---
        'Sample tags': {
            'type': 'tags',
            'tooltip': 'A sample tags selector.',
            'displayOrder': 7,
        },
        # --- Layer type ---
        'Sample layer': {
            'type': 'layer',
            'required': False,
            'tooltip': 'A sample layer selector.',
            'displayOrder': 8,
        },
        # --- Batch fields (standard pattern from cellposesam) ---
        'Batch XY': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'XY positions to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 9,
        },
        'Batch Z': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Z slices to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 10,
        },
        'Batch Time': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Time points to iterate over',
                'persistentPlaceholder': True,
                'filled': True,
            },
            'displayOrder': 11,
        },
    }
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    worker = WorkerClient(datasetId, apiUrl, token, params)

    # Read all interface values to demonstrate access patterns
    sample_number = float(worker.workerInterface.get('Sample number', 42))
    sample_text = worker.workerInterface.get('Sample text', '')
    sample_select = worker.workerInterface.get('Sample select', 'Option A')
    sample_checkbox = worker.workerInterface.get('Sample checkbox', False)

    tile_width = worker.datasetClient.tiles['tileWidth']
    tile_height = worker.datasetClient.tiles['tileHeight']

    # Demonstrate progress messaging
    sendProgress(0.1, 'Starting sample interface worker',
                 f'Number={sample_number}, Text="{sample_text}", Select={sample_select}')
    time.sleep(1)

    # Demonstrate warning messaging (without info)
    sendWarning('This is a sample warning')
    time.sleep(1)

    # Demonstrate warning messaging (with info)
    sendWarning('This is a sample warning with info',
                info='This warning is intentional and demonstrates the sendWarning() function.')
    time.sleep(1)

    # Demonstrate error messaging (without info)
    sendError('This is a sample error')
    time.sleep(1)

    # Demonstrate error messaging (with info)
    sendError('This is a sample error with info',
              info='This error is intentional and demonstrates the sendError() function. '
                   'The worker will continue running after this.')
    time.sleep(1)

    # Demonstrate conditional behavior based on checkbox
    if sample_checkbox:
        sendWarning('Checkbox was checked',
                    info='You checked the sample checkbox, so this extra warning was sent.')

    sendProgress(0.5, 'Generating sample annotations',
                 'Creating a small number of random squares to demonstrate annotation creation.')

    # Use WorkerClient batch mode to create a few random squares
    half = sample_number / 2.0  # Use the number field as square size

    def generate_sample_squares(image):
        """Generate a small number of sample squares to demonstrate the annotation pipeline."""
        polygons = []
        for _ in range(5):
            cx = random.uniform(half, max(tile_width - half, half + 1))
            cy = random.uniform(half, max(tile_height - half, half + 1))
            square = [
                (cx - half, cy - half),
                (cx + half, cy - half),
                (cx + half, cy + half),
                (cx - half, cy + half),
            ]
            polygons.append(square)
        return polygons

    worker.process(generate_sample_squares, f_annotation='polygon',
                   progress_text='Running sample interface worker')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample interface worker demonstrating all interface types and messaging')

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
