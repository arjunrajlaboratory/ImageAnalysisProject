import argparse
import json
import sys

import annotation_client.workers as workers


def compute(datasetId, apiUrl, token, params):
    """
    In this function:
    Annotation workers create new annotations
    Property workers compute annotation properties
    """
    print(datasetId)
    print(apiUrl)
    print(token)
    print(params)

    # The values retrieved using the worker interface are always in a dict: params['workerInterface']

    # WARNING: accessing value was done using params['workerInterface']['My property']['value'],
    # it is now done using params['workerInterface']['My property'] as the typescript type of
    # params['workerInterface'] is IWorkerInterfaceValues
    print(params['workerInterface'])


def interface(image):
    """
    Send parameters required by compute() to the frontend
    See WorkerInterfaceValues.vue and the type IWorkerInterface in model.ts to be updated on what can be requested
    """
    workerInterface = {
        'My number property': {
            'type': 'number',
            'min': -1,
            'max': 5,
            'default': 3.14,
        },
        'My text property': {
            'type': 'text',
            'default': 'Hello, world!',
        },
        'My tags property': {
            'type': 'tags',
            'default': ['cell', 'nuclei'],
        },
        'My layer property': {
            'type': 'layers',
            'default': 0,
        },
        'My select property': {
            'type': 'select',
            'default': 'blue',
            'items': ['orange', 'blue', 'green'],
        },
        'My channel property': {
            'type': 'channel',
            'default': 3,
        },
    }
    # Send the interface object to the server
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    client.setWorkerImageInterface(image, workerInterface)


def preview():
    # TODO: I don't know what preview should do
    pass


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

    # Params is:
    #   - for request 'compute':
    #       see 'computeAnnotationWithWorker' of AnnotationsAPI.ts for annotation workers
    #       see 'computeProperty' of PropertiesAPI.ts for property workers
    #   - for request 'interface':
    #       params is { 'image': image }
    #   - for request 'preview':
    #       see 'requestWorkerPreview' of PropertiesAPI.ts
    params = json.loads(args.parameters)
    datasetId = args.datasetId
    apiUrl = args.apiUrl
    token = args.token

    # As jammy is the base docker image, python version is >= 3.10.4
    # We can use match/case
    match args.request:
        case 'compute':
            compute(datasetId, apiUrl, token, params)
        case 'interface':
            interface(params['image'])
        case 'preview':
            preview()
