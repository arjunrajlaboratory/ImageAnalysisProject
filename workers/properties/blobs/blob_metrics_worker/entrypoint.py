import argparse
import json
import sys

import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendWarning

import annotation_utilities.annotation_tools as annotation_tools
# import annotation_utilities.units as units # Preserved for later use
from shapely.geometry import Polygon
import numpy as np


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Blob Metrics': {
            'type': 'notes',
            'value': 'This tool computes a variety of metrics for the objects in the specified channel. '
                     'The metrics computed are: area, perimeter, centroid, compactness, elongation, convexity, solidity, '
                     'rectangularity, circularity, fractal dimension, and eccentricity.',
            'displayOrder': 0,
        },
        'Use physical units': {
            'type': 'checkbox',
            'value': False,
            'tooltip': 'If checked, the metrics will be computed in physical units. If unchecked, the metrics will be computed in pixels.',
            'displayOrder': 1,
        },
        'Units': {
            'type': 'select',
            'items': ['m', 'mm', 'µm', 'nm'],
            'default': 'µm',
            'tooltip': 'The units of the physical units. If unchecked, the metrics will be computed in pixels.',
            'displayOrder': 2,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def convert_units(pixelSize, to_units):
    """
    Convert a pixel size to a different unit.
    """
    current_units = pixelSize['unit']
    current_value = pixelSize['value']
    if current_units == 'm':
        current_value *= 1
    elif current_units == 'mm':
        current_value *= 1e-3
    elif current_units == 'µm':
        current_value *= 1e-6
    elif current_units == 'nm':
        current_value *= 1e-9
    else:
        raise ValueError(f"Unknown unit: {current_units}")

    if to_units == 'm':
        current_value *= 1
    elif to_units == 'mm':
        current_value *= 1e3
    elif to_units == 'µm':
        current_value *= 1e6
    elif to_units == 'nm':
        current_value *= 1e9
    else:
        raise ValueError(f"Unknown unit: {to_units}")

    return {'unit': to_units, 'value': current_value}


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

    use_physical_units = params.get(
        'workerInterface', {}).get('Use physical units', False)
    final_units = params.get('workerInterface', {}).get('Units', 'µm')

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

    # We need at least one annotation
    if len(annotationList) == 0:
        sendWarning('No annotations found',
                    info='No annotations with the specified tags and shape were found.')
        return

    pixelSize = params['scales']['pixelSize']
    print(f"The pixel size is: {pixelSize}")
    # If the pixel size is 0, we can't compute physical units
    if pixelSize.get('value', 0) == 0:
        sendWarning('No pixel size found in the configuration')
        use_physical_units = False

    if use_physical_units:
        # TODO: Once we have a unit package, we can use it here as units.convert_units
        pixelSize = convert_units(pixelSize, final_units)
        pixel_length = pixelSize['value']
    else:
        pixel_length = 1

    number_annotations = len(annotationList)
    property_value_dict = {}  # Initialize as a dictionary
    for i, annotation in enumerate(annotationList):

        polygon_coords = [list(coordinate.values())[0:2]
                          for coordinate in annotation['coordinates']]
        if len(polygon_coords) < 3:
            sendWarning("Incorrect polygon detected",
                        info="Polygon with less than 3 points found.")
            continue
        poly = Polygon(polygon_coords)
        convex_hull = poly.convex_hull
        min_rect = poly.minimum_rotated_rectangle

        # Calculate elongation
        min_rect_coords = np.array(min_rect.exterior.coords)
        edges = np.diff(min_rect_coords, axis=0)
        length = max(np.linalg.norm(edges, axis=1))
        width = min(np.linalg.norm(edges, axis=1))
        elongation = 1 - (width / length)

        # Calculate eccentricity
        coords = np.array(poly.exterior.coords)
        cent = poly.centroid
        centered_coords = coords - [cent.x, cent.y]
        inertia_tensor = np.dot(centered_coords.T, centered_coords)
        eigenvalues = np.linalg.eigvals(inertia_tensor)
        eccentricity = np.sqrt(1 - (min(eigenvalues) / max(eigenvalues)))

        # Helper function to safely compute ratios
        def safe_compute(calculation):
            try:
                result = float(calculation)
                return result if np.isfinite(result) else None
            except:
                return None

        prop = {
            'Area': float(poly.area) * pixel_length ** 2,
            'Perimeter': float(poly.length) * pixel_length,
            'Centroid': {'x': float(poly.centroid.x) * pixel_length, 'y': float(poly.centroid.y) * pixel_length},
            'Compactness': safe_compute(4 * np.pi * poly.area / (poly.length ** 2)),
            'Elongation': elongation,
            'Convexity': safe_compute(poly.area / convex_hull.area),
            'Solidity': safe_compute(poly.length / convex_hull.length),
            'Rectangularity': safe_compute(poly.area / (length * width)),
            'Circularity': safe_compute(4 * np.pi * poly.area / (poly.length ** 2)),
            'Fractal_Dimension': safe_compute(2 * np.log(poly.length) / np.log(poly.area)),
            'Eccentricity': eccentricity
        }
        # Add prop to the dictionary with annotation['_id'] as the key
        property_value_dict[annotation['_id']] = prop
        # Only send progress every number_annotations / 100
        if number_annotations > 100:
            if i % int(number_annotations / 100) == 0:
                sendProgress((i+1)/number_annotations, 'Computing blob metrics',
                             f"Processing annotation {i+1}/{number_annotations}")
        else:
            sendProgress((i+1)/number_annotations, 'Computing blob metrics',
                         f"Processing annotation {i+1}/{number_annotations}")

    dataset_property_value_dict = {datasetId: property_value_dict}

    # Only send the metrics if we have at least one valid annotation
    if len(property_value_dict) > 0:
        sendProgress(0.5, 'Done computing',
                     'Sending computed metrics to the server')
        workerClient.add_multiple_annotation_property_values(
            dataset_property_value_dict)
    else:
        sendWarning('No valid annotations',
                    info='No valid annotations were found to compute metrics on.')


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
