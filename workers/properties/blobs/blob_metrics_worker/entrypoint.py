import argparse
import json
import sys

import annotation_client.workers as workers
from annotation_client.utils import sendProgress

import annotation_utilities.annotation_tools as annotation_tools

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
    # Here's an example of what the "params" dict might look like:
    # {'id': '65bc10b3e62fc888551f168d', 'name': 'metrics2', 'image': 'properties/blob_metrics:latest', 'tags': {'exclusive': False, 'tags': ['nucleus']}, 'shape': 'polygon', 'workerInterface': {}, 'scales': {'pixelSize': {'unit': 'mm', 'value': 0.000219080212825376}, 'tStep': {'unit': 's', 'value': 1}, 'zStep': {'unit': 'm', 'value': 1}}}
    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    print(f"Found {len(annotationList)} annotations with shape 'polygon'")
    print(f"The tags are: {params.get('tags', {}).get('tags', [])}")
    print(f"The exclusive flag is: {params.get('tags', {}).get('exclusive', False)}")
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, params.get('tags', {}).get('tags', []), params.get('tags', {}).get('exclusive', False))
    print(f"Found {len(annotationList)} annotations with the specified tags")

    # We need at least one annotation
    if len(annotationList) == 0:
        return


    number_annotations = len(annotationList)
    property_value_dict = {}  # Initialize as a dictionary
    for i, annotation in enumerate(annotationList):

        polygon_coords = [list(coordinate.values())[0:2] for coordinate in annotation['coordinates']]
        poly = Polygon(polygon_coords)

        prop = {
            'Area': float(poly.area),
            'Perimeter': float(poly.length),
            'Centroid': {'x': float(poly.centroid.x), 'y': float(poly.centroid.y)}
        }
        # Add prop to the dictionary with annotation['_id'] as the key
        property_value_dict[annotation['_id']] = prop
        sendProgress((i+1)/number_annotations, 'Computing blob metrics', f"Processing annotation {i+1}/{number_annotations}")

    dataset_property_value_dict = {datasetId: property_value_dict}

    workerClient.add_multiple_annotation_property_values(dataset_property_value_dict)

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
