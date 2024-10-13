import argparse
import json
import sys

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

from functools import partial

import annotation_client.workers as workers

from piscis import Piscis
from piscis.paths import MODELS_DIR

from worker_client import WorkerClient

import utils


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    models = sorted(path.stem for path in MODELS_DIR.glob('*'))
    girder_models = [model['name'] for model in utils.list_girder_models(client.client)[0]]
    models = sorted(list(set(models + girder_models)))

    # Available types: number, text, tags, layer
    interface = {
        'Piscis': {
            'type': 'notes',
            'value': 'This tool uses the Piscis model to find points in images. '
                     'It can be used to segment in 2D or 3D.',
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
        'Select a model': {
            'type': 'notes',
            'value': 'Select the model to use for segmentation. These can be pre-trained models or models you have generated yourself with Piscis Train. '
                     'The model determines how sensitive the point detection is.',
            'displayOrder': 4,
        },
        'Model': {
            'type': 'select',
            'items': models,
            'default': '20230905',
            'displayOrder': 5,
        },
        'Mode for z-stack usage': {
            'type': 'notes',
            'value': 'If you want to segment in 3D, select "Z-Stack". Otherwise, if you just want to segment in each z-slice individually, select "Current Z". '
                     'If you select "Z-Stack", then the model will be used to segment all z-slices at once, and it will ignore the "Batch Z" field.',
            'displayOrder': 6,
        },
        'Mode': {
            'type': 'select',
            'items': ['Current Z', 'Z-Stack'],
            'default': 'Current Z',
            'displayOrder': 7,
        },
        'Scale parameter': {
            'type': 'notes',
            'value': 'This parameter controls the size of the objects that are detected. '
                     'It is a multiplier on the size of the objects in the model.',
            'displayOrder': 8,
        },
        'Scale': {
            'type': 'number',
            'min': 0,
            'max': 5,
            'default': 1,
            'displayOrder': 9,
        },
        'Threshold note': {
            'type': 'notes',
            'value': 'The threshold parameter honestly does not change much; use a different model if you need to change specificity.',
            'displayOrder': 10,
        },
        'Threshold': {
            'type': 'number',
            'min': 0,
            'max': 9,
            'default': 1.0,
            'displayOrder': 11,
        },
        'Notes on skipping frames': {
            'type': 'notes',
            'value': 'Sometimes you may want to skip processing frames that do not have any objects of a particular tag. '
                     'If empty, all frames will be processed.',
            'displayOrder': 12,
        },
        'Skip Frames Without': {
            'type': 'tags',
            'displayOrder': 13,
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

    gc = worker.annotationClient.client
    utils.download_girder_model(gc, model_name)
    utils.download_girder_cache(gc, mode='predict')

    model = Piscis(model_name=model_name, batch_size=1)
    f_process = partial(run_model, model=model, stack=stack, scale=scale, threshold=threshold)
    worker.process(f_process, f_annotation='point', stack_zs='all' if stack else None, progress_text='Running Piscis')

    utils.upload_girder_cache(gc, mode='predict')


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
