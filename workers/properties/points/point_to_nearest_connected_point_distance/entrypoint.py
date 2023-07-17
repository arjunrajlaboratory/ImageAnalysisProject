import argparse
import json
import sys

import annotation_client.workers as workers
import annotation_client.annotations as annotations
import annotation_tools

import cv2 as cv
import numpy as np
import math

def calculate_distance(point1, point2):
    x1, y1, z1 = point1['coordinates'][0]['x'], point1['coordinates'][0]['y'], point1['coordinates'][0]['z']
    x2, y2, z2 = point2['coordinates'][0]['x'], point2['coordinates'][0]['y'], point2['coordinates'][0]['z']
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def find_matching_annotations_by_location(source, target_list, Time=True, XY=True, Z=True):
    """
    This function filters the target_list based on the 'location' of the source point.
    The function parameters 'Time', 'XY', and 'Z' can be set to True or False to specify whether these 'location' attributes need to be matched.
    By default, all of these parameters are set to True, meaning all 'location' attributes need to match.

    Parameters:
    source (dict): The source point annotation object
    target_list (list): The list of target point annotation objects
    Time (bool): Specifies whether the 'Time' attribute of 'location' needs to be matched. Default is True.
    XY (bool): Specifies whether the 'XY' attribute of 'location' needs to be matched. Default is True.
    Z (bool): Specifies whether the 'Z' attribute of 'location' needs to be matched. Default is True.

    Returns:
    list: The filtered list of target point annotation objects that match the specified 'location' attributes

    Example of usage:
    1) Matching all 'location' attributes:
    source = {...}  # source point annotation object
    target_list = [...]  # target point annotation list
    matching_annotations = find_matching_annotations_by_location(source, target_list)

    2) Matching specified 'location' attributes (in this case, 'Time' and 'XY'):
    source = {...}  # source point annotation object
    target_list = [...]  # target point annotation list
    matching_annotations = find_matching_annotations_by_location(source, target_list, Time=True, XY=True, Z=False)
    """
    params = {'Time': Time, 'XY': XY, 'Z': Z}
    return [target for target in target_list if all(source['location'].get(attr) == target['location'].get(attr) for attr, value in params.items() if value)]


def find_closest_point(source, target_list):
    min_distance = float('inf')
    closest_point = None
    for target in target_list:
        if source['_id'] == target['_id']:
            continue
        distance = calculate_distance(source, target)
        if distance < min_distance:
            min_distance = distance
            closest_point = target
    return closest_point, min_distance

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Tags of points to measure distance to': {
            'type': 'tags',
            'required': True,
        },
        'Target tag match': {
            'type': 'select',
            'items': ['Any', 'Exact'],
            'default': 'Exact',
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
    def get_child_annotations_from_connections(connections):
        child_annotations = []
        for connection in connections:
            childId = connection.get('childId')
            if childId:
                child_annotation = annotationClient.getAnnotationById(childId)
                child_annotations.append(child_annotation)
        return child_annotations

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    # First, let's get a list of all point annotations
    annotationList = workerClient.get_annotation_list_by_shape('point', limit=0)
    
    # Source points here, filtered by main instantiating interface
    filteredPointList = annotation_tools.get_annotations_with_tags(annotationList,params['tags']['tags'],exclusive=params['tags']['exclusive'])

    # Target points here, filtered by worker interface
    workerInterface = params['workerInterface']
    tags = set(workerInterface.get('Tags of points to measure distance to', None))
    exclusive = workerInterface['Target tag match'] == 'Exact'
    #targetPointList = annotation_tools.get_annotations_with_tags(annotationList,tags,exclusive=exclusive)

    # We need at least one annotation
    if len(filteredPointList) == 0:
        return

    for source in filteredPointList:
        connections = annotationClient.getAnnotationConnections(datasetId,parentId=source['_id'])
        targetPointList = get_child_annotations_from_connections(connections)
        targetPointList = annotation_tools.get_annotations_with_tags(targetPointList,tags,exclusive=exclusive)
        
        # Add the attributes you want to match as keyword arguments here. If no arguments are provided, all 'location' attributes will be matched.
        closest_point, min_distance = find_closest_point(source, targetPointList)
        
        if min_distance != float('inf'):
            workerClient.add_annotation_property_values(source, min_distance)
            #print(min_distance) #workerClient.add_annotation_property_values(source, min_distance)


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute the distance between one set of points and another')

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
