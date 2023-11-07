import argparse
import json
import sys

import annotation_client.workers as workers

import numpy as np
# from point_in_polygon import point_in_polygon
from annotation_utilities.point_in_polygon import point_in_polygon


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Tags': {
            'type': 'tags'
        },
        'Exclusive': {
            'type': 'select',
            'items': ['Yes', 'No'],
            'default': 'Yes'
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

    workerInterface = params['workerInterface']
    tags = set(workerInterface.get('Tags', None))
    exclusive = workerInterface['Exclusive'] == 'Yes'

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    pointList = workerClient.get_annotation_list_by_shape('point', limit=0)

    filteredPointList = []
    for point in pointList:
        point_tags = set(point['tags'])
        if (exclusive and (point_tags == tags)) or ((not exclusive) and (len(point_tags & tags) > 0)):
            filteredPointList.append(point)

    points = np.array([[point['location'][i]
                        for i in ['Time', 'XY', 'Z']] + list(point['coordinates'][0].values())[1::-1]
                       for point in filteredPointList])

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    for annotation in annotationList:

        image = workerClient.get_image_for_annotation(annotation)

        if image is None:
            continue

        polygon = np.array([[coordinate[i] for i in ['y', 'x']] for coordinate in annotation['coordinates']])
        filtered_points = points[np.all(points[:, :3] == np.array([annotation['location'][i] for i in ['Time', 'XY', 'Z']]), axis=1)][:, -2:]
        point_count = np.sum(point_in_polygon(filtered_points, polygon))

        workerClient.add_annotation_property_values(annotation, int(point_count))


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
