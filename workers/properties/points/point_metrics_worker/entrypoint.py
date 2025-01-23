import argparse
import json
import sys

import annotation_client.workers as workers
from annotation_client.utils import sendProgress
import annotation_utilities.annotation_tools as annotation_tools


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    interface = {
        'Point Metrics': {
            'type': 'notes',
            'value': 'This tool adds a property to the points to document their coordinates. '
                     'It gives each point an x and y coordinate.',
            'displayOrder': 0,
        }
    }
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
    annotationList = workerClient.get_annotation_list_by_shape(
        'point', limit=0)
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, params.get(
        'tags', {}).get('tags', []), params.get('tags', {}).get('exclusive', False))

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    number_annotations = len(annotationList)
    property_value_dict = {}  # Initialize as a dictionary
    for i, annotation in enumerate(annotationList):

        x = annotation['coordinates'][0]['x']
        y = annotation['coordinates'][0]['y']

        prop = {
            'x': float(x),
            'y': float(y)
        }
        # Add prop to the dictionary with annotation['_id'] as the key
        property_value_dict[annotation['_id']] = prop
        # Only send progress every number_annotations / 100
        if number_annotations > 100:
            if i % int(number_annotations / 100) == 0:
                sendProgress((i+1)/number_annotations, 'Computing point metrics',
                             f"Processing annotation {i+1}/{number_annotations}")
        else:
            sendProgress((i+1)/number_annotations, 'Computing point metrics',
                         f"Processing annotation {i+1}/{number_annotations}")

    dataset_property_value_dict = {datasetId: property_value_dict}

    sendProgress(0.5, 'Done computing',
                 'Sending computed metrics to the server')
    workerClient.add_multiple_annotation_property_values(
        dataset_property_value_dict)


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
