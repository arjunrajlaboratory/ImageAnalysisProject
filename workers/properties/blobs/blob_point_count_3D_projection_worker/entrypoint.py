import argparse
import json
import sys

import annotation_client.workers as workers

#from point_in_polygon import point_in_polygon
from shapely.geometry import Point, Polygon

from annotation_utilities.point_in_polygon import point_in_polygon
from annotation_utilities import annotation_tools

#import annotation_tools
from rtree import index

import numpy as np


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
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
    tags = set(workerInterface.get('Tags of points to count', None))
    exclusive = workerInterface['Exact tag match?'] == 'Yes'

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    blobAnnotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    pointList = workerClient.get_annotation_list_by_shape('point', limit=0)

    # We need at least one annotation
    if len(blobAnnotationList) == 0:
        return

    filteredPointList = annotation_tools.get_annotations_with_tags(pointList,tags,exclusive=exclusive)

    counts = [0] * len(blobAnnotationList) # We don't need to save counts for every polygon, but whatever.

    # Initialize variables for tracking changes in time_value and xy_value
    previous_time_value = None
    previous_xy_value = None

    # Initialize index and filtered_points list
    idx = index.Index()
    filtered_points = []
    myPoints = []

    for i, blob in enumerate(blobAnnotationList):
        # Extract the x and y coordinates from the dictionary object and create a list of tuples
        xy_coords = [(coord['x'], coord['y']) for coord in blob['coordinates']]

        # Create a `Polygon` object from the x and y coordinate tuples
        polygon = Polygon(xy_coords)
        
        # Extract the Time and XY values from the blob annotation
        time_value = blob['location']['Time']
        xy_value = blob['location']['XY']

        # If time_value or xy_value have changed, update the filtered_points and rebuild the index
        if time_value != previous_time_value or xy_value != previous_xy_value:
            filtered_points = annotation_tools.filter_elements_T_XY(filteredPointList, time_value, xy_value)
            
            # Rebuild the index
            idx = index.Index()
            myPoints = annotation_tools.create_points_from_annotations(filtered_points)
            for j, point in enumerate(myPoints):
                idx.insert(j, point.bounds)
        
        for j in idx.intersection(polygon.bounds):
            point = myPoints[j]
            if polygon.contains(point):
                counts[i] += 1
                
        # Update previous_time_value and previous_xy_value for the next iteration
        previous_time_value = time_value
        previous_xy_value = xy_value

        print(f"Number of points within polygon {i}: {counts[i]}")
        workerClient.add_annotation_property_values(blob, int(counts[i]))


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
