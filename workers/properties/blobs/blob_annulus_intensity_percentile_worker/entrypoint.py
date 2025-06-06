import argparse
import json
import sys
import timeit


import annotation_client.workers as workers
from annotation_client.utils import sendProgress
import annotation_client.tiles as tiles

import annotation_utilities.annotation_tools as annotation_tools

import numpy as np
from skimage import draw
from skimage import morphology
from collections import defaultdict
from shapely.geometry import Polygon


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Blob Annulus Intensity Percentile': {
            'type': 'notes',
            'value': 'This tool computes the pixel intensity in an annulus around the objects in the specified channel at the specified percentile. '
                     'For instance, if you set the percentile to 90, it will compute the 90th percentile intensity of the annular region. '
                     'The size of the annulus is defined by the radius.',
            'displayOrder': 0,
        },
        'Channel': {
            'type': 'channel',
            'required': True,
            'tooltip': 'Compute pixel intensities in this channel.\n'
                       'The channel does not have to be the same as the layer the annotations are on.',
            'displayOrder': 1,
        },
        'Radius': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10,
            'units': 'pixels',
            'displayOrder': 2,
        },
        'Percentile': {
            'type': 'number',
            'min': 0,
            'max': 99.99999,
            'default': 50,
            'displayOrder': 3,
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

    workerClient = workers.UPennContrastWorkerClient(
        datasetId, apiUrl, token, params)
    annotationList = workerClient.get_annotation_list_by_shape(
        'polygon', limit=0)
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, params.get(
        'tags', {}).get('tags', []), params.get('tags', {}).get('exclusive', False))

    channel = params['workerInterface']['Channel']
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)
    annulus_radius = float(params['workerInterface']['Radius'])
    percentile = float(params['workerInterface']['Percentile'])

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    start_time = timeit.default_timer()

    grouped_annotations = defaultdict(list)
    for annotation in annotationList:
        location_key = (annotation['location']['Time'],
                        annotation['location']['Z'], annotation['location']['XY'])
        grouped_annotations[location_key].append(annotation)

    number_annotations = len(annotationList)

    # For reporting progress
    processed_annotations = 0

    property_value_dict = {}  # Initialize as a dictionary

    for location_key, annotations in grouped_annotations.items():
        time, z, xy = location_key
        frame = datasetClient.coordinatesToFrameIndex(xy, z, time, channel)
        image = datasetClient.getRegion(datasetId, frame=frame)
        image = image.squeeze()

        if image is None:
            continue

        # Compute properties for all annotations at that location
        for annotation in annotations:
            polygon = np.array([[coordinate['y'] - 0.5, coordinate['x'] - 0.5]
                                for coordinate in annotation['coordinates']])

            if len(polygon) < 3:  # Skip if the polygon is not valid
                continue

            rr, cc = draw.polygon(
                polygon[:, 0], polygon[:, 1], shape=image.shape)
            original_coords = set(zip(rr, cc))

            # Get coordinates of dilated polygon
            dilated_polygon = Polygon(polygon).buffer(annulus_radius)
            rr_dilated, cc_dilated = draw.polygon(
                np.array(dilated_polygon.exterior.coords)[:, 0],
                np.array(dilated_polygon.exterior.coords)[:, 1],
                shape=image.shape
            )
            dilated_coords = set(zip(rr_dilated, cc_dilated))

            # Get just the annulus coordinates (in dilated but not in original)
            annulus_coords = dilated_coords - original_coords

            # Skip if there are no pixels in the annulus
            if len(annulus_coords) == 0:
                continue

            rr_annulus, cc_annulus = zip(*annulus_coords)
            intensities = image[rr_annulus, cc_annulus]

            if len(intensities) == 0:  # Skip if there are no pixels in the mask
                continue

            if intensities.size > 0:
                # Calculating the desired metrics

                percentile_intensity = np.percentile(intensities, percentile)
                prop_name = f'{percentile}thPercentileIntensity'

                prop = {
                    prop_name: float(percentile_intensity),
                }

                property_value_dict[annotation['_id']] = prop

            processed_annotations += 1
            # Only send progress every number_annotations / 100
            if number_annotations > 100:
                if processed_annotations % int(number_annotations / 100) == 0:
                    sendProgress(processed_annotations / number_annotations, 'Computing blob intensity',
                                 f"Processing annotation {processed_annotations}/{number_annotations}")
            else:
                sendProgress(processed_annotations / number_annotations, 'Computing blob intensity',
                             f"Processing annotation {processed_annotations}/{number_annotations}")

    dataset_property_value_dict = {datasetId: property_value_dict}

    sendProgress(0.5, 'Done computing',
                 'Sending computed metrics to the server')
    workerClient.add_multiple_annotation_property_values(
        dataset_property_value_dict)

    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Executed the code in: {execution_time} seconds")


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
