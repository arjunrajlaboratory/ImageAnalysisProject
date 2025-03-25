import argparse
import json
import sys
import time

import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendWarning, sendError

import annotation_utilities.annotation_tools as annotation_tools
# import annotation_utilities.units as units # Preserved for later use
from shapely.geometry import Polygon
import numpy as np


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Test worker': {
            'type': 'notes',
            'value': 'This tool is a test worker. It does not compute any metrics.',
            'displayOrder': 0,
        },
        'Test checkbox': {
            'type': 'checkbox',
            'value': False,
            'tooltip': 'This is a test checkbox.',
            'displayOrder': 1,
        },
        'Test select': {
            'type': 'select',
            'items': ['Item 1', 'Item 2', 'Item 3'],
            'default': 'Item 1',
            'tooltip': 'This is a test select.',
            'displayOrder': 2,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    """
    Params is a dict containing the following parameters:
    required:
        name: The name of the property
        id: The id of the property
        propertyType: can be "morphology", "relational", or "layer"
    optional:
        annotationId: A list of annotation ids for which the property should be computed
        shape: The shape of annotations that should be used
        layer: Which specific layer should be used for intensity calculations
        tags: A list of annotation tags, used when counting for instance the number of connections to specific tagged annotations
    """

    workerClient = workers.UPennContrastWorkerClient(
        datasetId, apiUrl, token, params)

    test_checkbox = params.get(
        'workerInterface', {}).get('Test checkbox', False)
    test_select = params.get('workerInterface', {}).get('Test select', 'Item 1')

    # Here's an example of what the "params" dict might look like:
    # {'id': '65bc10b3e62fc888551f168d', 'name': 'metrics2', 'image': 'properties/blob_metrics:latest', 'tags': {'exclusive': False, 'tags': ['nucleus']}, 'shape': 'polygon', 'workerInterface': {}, 'scales': {'pixelSize': {'unit': 'mm', 'value': 0.000219080212825376}, 'tStep': {'unit': 's', 'value': 1}, 'zStep': {'unit': 'm', 'value': 1}}}
    annotationList = workerClient.get_annotation_list_by_shape(
        'polygon', limit=0)
    print(f"Found {len(annotationList)} annotations with shape 'polygon'")
    print(f"The tags are: {params.get('tags', {}).get('tags', [])}")
    # print(f"The exclusive flag is: {params.get('tags', {}).get('exclusive', False)}")
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, params.get(
        'tags', {}).get('tags', []), params.get('tags', {}).get('exclusive', False))
    print(f"Found {len(annotationList)} annotations with the specified tags")

    sendProgress(0.5, 'Starting test worker',
                 'This is a test worker. It does not compute any metrics.')

    time.sleep(2)

    sendProgress(0.8, 'Next step',
                 'This is the next step of the test worker.')

    time.sleep(2)

    sendWarning('This is a warning',
                info='This is an info message.')

    time.sleep(2)

    sendError('This is an error',
              info='This is an info message.')

    sendProgress(0.9, 'Final step',
                 'This is the final step of the test worker.')

    time.sleep(2)

    sendProgress(1.0, 'Done',
                 'This is the final step of the test worker.')


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

    match args.request:
        case 'compute':
            compute(datasetId, apiUrl, token, params)
        case 'interface':
            interface(params['image'], apiUrl, token)
