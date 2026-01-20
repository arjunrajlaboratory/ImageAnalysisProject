import utils
from worker_client import WorkerClient
from piscis.paths import MODELS_DIR
from piscis import Piscis
import annotation_client.workers as workers
from functools import partial
import argparse
import json
import sys
import torch

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    models = sorted(path.stem for path in MODELS_DIR.glob('*'))
    girder_models = [model['model_name'] for model in utils.list_girder_models(client.client)[0]]
    models = sorted(list(set(models + girder_models)))

    # Available types: number, text, tags, layer
    interface = {
        'Piscis': {
            'type': 'notes',
            'value': 'This tool uses the Piscis model to find points in images. '
                     'It can be used to segment in 2D or 3D. '
                     '<a href="https://docs.nimbusimage.com/documentation/analyzing-image-data-with-objects-connections-and-properties/tools-for-making-objects#piscis-for-automated-spot-finding" target="_blank">Learn more</a>',
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
            'default': '20251212',
            'tooltip': 'Select the model to use for segmentation. These can be pre-trained models or\nmodels you have generated yourself with Piscis Train. '
                       'The model determines how sensitive the point detection is.',
            'noCache': True,
            'displayOrder': 5,
        },
        'Mode': {
            'type': 'select',
            'items': ['Current Z', 'Z-Stack'],
            'default': 'Current Z',
            'tooltip': 'If you want to segment in 3D, select "Z-Stack". Otherwise, if you just want to segment in each z-slice individually, select "Current Z".\n'
                       'If you select "Z-Stack", then the model will be used to segment all z-slices at once, and it will ignore the "Batch Z" field.',
            'displayOrder': 7,
        },
        'Scale': {
            'type': 'number',
            'min': 0,
            'max': 5,
            'default': 1,
            'tooltip': 'This parameter controls the size of the objects that are detected.\n'
                       'It is a multiplier on the size of the objects in the model.',
            'displayOrder': 9,
        },
        'Threshold': {
            'type': 'number',
            'min': 0,
            'max': 9,
            'default': 0.5,
            'tooltip': 'The threshold parameter honestly does not change much.\nUse a different model if you need to change specificity.',
            'displayOrder': 11,
        },
        'Skip Frames Without': {
            'type': 'tags',
            'tooltip': 'Sometimes you may want to skip processing frames that do not have any objects of a particular tag.\n'
                       'If empty, all frames will be processed.',
            'displayOrder': 13,
        }
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def run_model(image, model, stack, scale, threshold):

    coords = model.predict(image, stack=stack, scale=scale,
                           threshold=threshold, intermediates=False)
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

    model = Piscis(model_name=model_name, batch_size=1, device='cuda' if torch.cuda.is_available() else 'cpu')
    f_process = partial(run_model, model=model, stack=stack, scale=scale, threshold=threshold)
    worker.process(f_process, f_annotation='point',
                   stack_zs='all' if stack else None, progress_text='Running Piscis')


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
