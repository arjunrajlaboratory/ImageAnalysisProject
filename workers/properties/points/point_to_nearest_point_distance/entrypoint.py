import argparse
import json
import sys

import annotation_client.workers as workers

import cv2 as cv
import numpy as np
import math

def calculate_distance(p1, p2):
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2 + (p2['z'] - p1['z'])**2)

def total_length(line):
    coordinates = line['coordinates']
    return sum(calculate_distance(coordinates[i], coordinates[i+1]) for i in range(len(coordinates)-1))

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Tags of points to measure distance to': {
            'type': 'tags'
        },
        'Target tag match': {
            'type': 'select',
            'items': ['Any', 'Exact'],
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

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    # First, let's get a list of all point annotations
    annotationList = workerClient.get_annotation_list_by_shape('point', limit=0)
    
    # Source points here, filtered by main instantiating interface
    filteredPointList = annotation_tools.get_annotations_with_tags(annotationList,params['tags']['tags'],exclusive=params['tags']['exclusive'])

    # Target points here, filtered by worker interface
    workerInterface = params['workerInterface']
    tags = set(workerInterface.get('Tags of points to count', None))
    exclusive = workerInterface['Target tag match'] == 'Exact'
    targetPointList = annotation_tools.get_annotations_with_tags(annotationList,tags,exclusive=exclusive)

    # We need at least one annotation
    if len(filteredPointList) == 0:
        return

    for annotation in filteredPointList:
        workerClient.add_annotation_property_values(annotation, total_length(annotation))


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute the x coordinate of the blob centroid')

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
