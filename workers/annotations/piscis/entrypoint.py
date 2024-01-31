import argparse
import json
import sys

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

from functools import partial

import annotation_client.workers as workers

from piscis import Piscis
from piscis.paths import MODELS_DIR

from worker_client import WorkerClient


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    models = sorted(path.stem for path in MODELS_DIR.glob('*'))

    # Available types: number, text, tags, layer
    interface = {
        'Model': {
            'type': 'select',
            'items': models,
            'default': models[-1]
        },
        'Mode': {
            'type': 'select',
            'items': ['Current Z', 'Z-Stack'],
            'default': 'Current Z'
        },
        'Scale': {
            'type': 'number',
            'min': 0,
            'max': 5,
            'default': 1
        },
        'Threshold': {
            'type': 'number',
            'min': 0,
            'max': 9,
            'default': 1.0
        },
        'Batch XY': {
            'type': 'text'
        },
        'Batch Z': {
            'type': 'text'
        },
        'Batch Time': {
            'type': 'text'
        },
        'Skip Frames Without': {
        	'type': 'tags'
        }
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def run_model(image, model, stack, scale, threshold):

    coords = model.predict(image, stack=stack, scale=scale, threshold=threshold, intermediates=False)
    coords[:, -2:] += 0.5

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
    model_name = worker.workerInterface['Model']
    stack = worker.workerInterface['Mode'] == 'Z-Stack'
    scale = float(worker.workerInterface['Scale'])
    threshold = float(worker.workerInterface['Threshold'])

    model = Piscis(model_name=model_name, batch_size=1)
    f_process = partial(run_model, model=model, stack=stack, scale=scale, threshold=threshold)

    worker.process(f_process, f_annotation='point', stack_z=stack, progress_text='Running Piscis')


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
