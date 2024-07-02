import argparse
import json
import sys
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point, Polygon

import annotation_client.workers as workers
import annotation_client.annotations as annotations
import annotation_utilities.annotation_tools as annotation_tools
from annotation_client.utils import sendProgress

def calculate_distance_to_blob(point, blob, distance_type='centroid'):
    point_coords = Point(point['coordinates'][0]['x'], point['coordinates'][0]['y'])
    blob_polygon = Polygon([(coord['x'], coord['y']) for coord in blob['coordinates']])
    
    if distance_type == 'centroid':
        return point_coords.distance(blob_polygon.centroid)
    elif distance_type == 'edge':
        return point_coords.distance(blob_polygon.boundary)
    else:
        raise ValueError("Invalid distance_type. Must be 'centroid' or 'edge'.")

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Blob tags': {
            'type': 'tags',
            'required': True,
        },
        'Distance type': {
            'type': 'select',
            'items': ['Centroid', 'Edge'],
            'default': 'Centroid',
        },
        'Create connection': {
            'type': 'checkbox',
            'default': False,
        },
    }
    client.setWorkerImageInterface(image, interface)

def compute(datasetId, apiUrl, token, params):
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)

    # Get point annotations
    pointList = workerClient.get_annotation_list_by_shape('point', limit=0)
    filteredPointList = annotation_tools.get_annotations_with_tags(pointList, params['tags']['tags'], exclusive=params['tags']['exclusive'])

    # Get blob annotations
    blobList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    blobTags = set(params['workerInterface']['Blob tags'])
    filteredBlobList = annotation_tools.get_annotations_with_tags(blobList, blobTags, exclusive=False)

    distance_type = params['workerInterface']['Distance type'].lower()
    create_connection = params['workerInterface']['Create connection']

    number_points = len(filteredPointList)
    property_value_dict = {}
    connections = []

    for i, point in enumerate(filteredPointList):
        min_distance = float('inf')
        nearest_blob = None

        for blob in filteredBlobList:
            if point['location'] == blob['location']:  # Check if point and blob are in the same location
                dist = calculate_distance_to_blob(point, blob, distance_type)
                if dist < min_distance:
                    min_distance = dist
                    nearest_blob = blob

        if nearest_blob:
            property_value_dict[point['_id']] = float(min_distance)

            if create_connection:
                connections.append({
                    'datasetId': datasetId,
                    'parentId': nearest_blob['_id'],
                    'childId': point['_id'],
                    'tags': list(set(point['tags'] + nearest_blob['tags']))
                })

        sendProgress((i+1)/number_points, 'Computing distances', f"Processing point {i+1}/{number_points}")

    dataset_property_value_dict = {datasetId: property_value_dict}

    sendProgress(0.9, 'Done computing', 'Sending computed distances to the server')
    workerClient.add_multiple_annotation_property_values(dataset_property_value_dict)

    if create_connection:
        sendProgress(0.95, 'Creating connections', 'Sending connections to the server')
        annotationClient.createMultipleConnections(connections)

    sendProgress(1.0, 'Finished', 'Worker completed successfully')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute distance from point to nearest blob')
    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str, required=True, action='store')

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