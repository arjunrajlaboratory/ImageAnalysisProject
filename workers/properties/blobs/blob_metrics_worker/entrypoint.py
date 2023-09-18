import argparse
import json
import sys

import annotation_client.workers as workers

import cv2 as cv
import numpy as np

from shapely.geometry import Polygon

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
    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    for annotation in annotationList:

        polygon_coords = [list(coordinate.values())[0:2] for coordinate in annotation['coordinates']]
        poly = Polygon(polygon_coords)

        prop = {
            'Area': float(poly.area),
            'Perimeter': float(poly.length),
            'Centroid': {'x': float(poly.centroid.x), 'y': float(poly.centroid.y)}
        }

        workerClient.add_annotation_property_values(annotation, prop)

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
