import argparse
import json
import sys

import annotation_client.workers as workers
from annotation_client.utils import sendProgress

from shapely.geometry import Point, Polygon

from annotation_utilities.point_in_polygon import point_in_polygon
from annotation_utilities import annotation_tools

from rtree import index

import numpy as np

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    interface = {
        'Tags of points to count': {
            'type': 'tags'
        },
        'Exact tag match?': {
            'type': 'select',
            'items': ['Yes', 'No'],
            'default': 'Yes'
        },
    }
    client.setWorkerImageInterface(image, interface)

def compute(datasetId, apiUrl, token, params):
    workerInterface = params['workerInterface']
    tags = set(workerInterface.get('Tags of points to count', None))
    exclusive = workerInterface['Exact tag match?'] == 'Yes'

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    
    sendProgress(0.1, 'Fetching data', 'Getting polygon annotations')
    blobAnnotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    
    sendProgress(0.3, 'Fetching data', 'Getting point annotations')
    pointList = workerClient.get_annotation_list_by_shape('point', limit=0)

    if len(blobAnnotationList) == 0:
        sendProgress(1.0, 'Complete', 'No polygon annotations found')
        return

    sendProgress(0.4, 'Processing data', 'Filtering point annotations')
    filteredPointList = annotation_tools.get_annotations_with_tags(pointList, tags, exclusive=exclusive)

    property_value_dict = {}
    previous_time_value = None
    previous_xy_value = None
    idx = index.Index()
    filtered_points = []
    myPoints = []

    total_blobs = len(blobAnnotationList)

    for i, blob in enumerate(blobAnnotationList):
        progress = 0.4 + (0.5 * (i / total_blobs))
        sendProgress(progress, 'Computing', f'Processing polygon {i+1} of {total_blobs}')

        xy_coords = [(coord['x'], coord['y']) for coord in blob['coordinates']]
        polygon = Polygon(xy_coords)
        
        time_value = blob['location']['Time']
        xy_value = blob['location']['XY']

        if time_value != previous_time_value or xy_value != previous_xy_value:
            filtered_points = annotation_tools.filter_elements_T_XY(filteredPointList, time_value, xy_value)
            
            idx = index.Index()
            myPoints = annotation_tools.create_points_from_annotations(filtered_points)
            for j, point in enumerate(myPoints):
                idx.insert(j, point.bounds)
        
        count = sum(1 for j in idx.intersection(polygon.bounds) if polygon.contains(myPoints[j]))
        
        property_value_dict[blob['_id']] = int(count)
        
        previous_time_value = time_value
        previous_xy_value = xy_value

    sendProgress(0.9, 'Saving data', 'Updating annotation property values')
    dataset_property_value_dict = {datasetId: property_value_dict}
    workerClient.add_multiple_annotation_property_values(dataset_property_value_dict)

    sendProgress(1.0, 'Complete', 'Property worker finished successfully')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute point counts within polygon annotations')

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