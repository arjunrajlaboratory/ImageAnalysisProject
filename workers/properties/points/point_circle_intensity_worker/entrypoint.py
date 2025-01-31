import argparse
import json
import sys
import time

import annotation_client.workers as workers
import annotation_client.tiles as tiles
import numpy as np
from skimage import draw
from collections import defaultdict

from annotation_client.utils import sendProgress
import annotation_utilities.annotation_tools as annotation_tools


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Point Intensity': {
            'type': 'notes',
            'value': 'This tool computes the average, maximum, minimum, median, total, 25th percentile, and 75th percentile intensities for all pixels in a circle around point annotations in the specified channel. '
                     'If you set the radius to 1 or smaller, then the intensity will be the pixel value at the point location. '
                     'If you set the radius to greater than 1, then the intensity will be the average of all pixels in the circle. '
                     'This can be helpful if you want to compute the background intensity in a region around the point, for instance.',
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
            'min': 0.5,
            'max': 10,
            'default': 1,
            'unit': 'pixels',
            'displayOrder': 2,
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

    # Constants
    radius = float(params['workerInterface']['Radius'])
    channel = params['workerInterface']['Channel']

    workerClient = workers.UPennContrastWorkerClient(
        datasetId, apiUrl, token, params)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)
    annotationList = workerClient.get_annotation_list_by_shape(
        'point', limit=0)
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, params.get(
        'tags', {}).get('tags', []), params.get('tags', {}).get('exclusive', False))

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    number_annotations = len(annotationList)

    grouped_annotations = defaultdict(list)
    for annotation in annotationList:
        location_key = (annotation['location']['Time'],
                        annotation['location']['Z'], annotation['location']['XY'])
        grouped_annotations[location_key].append(annotation)

    # For reporting progress
    processed_annotations = 0

    property_value_dict = {}  # Initialize as a dictionary

    for location_key, annotations in grouped_annotations.items():
        t, z, xy = location_key
        frame = tileClient.coordinatesToFrameIndex(xy, z, t, channel)
        image = tileClient.getRegion(datasetId, frame=frame)

        if image is None:
            continue

        for annotation in annotations:

            # image = workerClient.get_image_for_annotation(annotation)

            if image is None:
                continue

            geojsPoint = annotation['coordinates'][0]
            # Subtract 0.5 to convert from pixel corner to pixel center for skimage.draw.disk
            point = np.array([geojsPoint['y']-0.5, geojsPoint['x']-0.5])

            rr, cc = draw.disk(point, radius, shape=image.shape)

            if rr.size > 0:  # If the circle catches at least one pixel
                intensities = image[rr, cc]
                # Calculating the desired metrics
                mean_intensity = np.mean(intensities)
                max_intensity = np.max(intensities)
                min_intensity = np.min(intensities)
                median_intensity = np.median(intensities)
                q25_intensity = np.percentile(intensities, 25)
                q75_intensity = np.percentile(intensities, 75)
                total_intensity = np.sum(intensities)

                prop = {
                    'MeanIntensity': float(mean_intensity),
                    'MaxIntensity': float(max_intensity),
                    'MinIntensity': float(min_intensity),
                    'MedianIntensity': float(median_intensity),
                    '25thPercentileIntensity': float(q25_intensity),
                    '75thPercentileIntensity': float(q75_intensity),
                    'TotalIntensity': float(total_intensity),
                }

                property_value_dict[annotation['_id']] = prop

                # Only send progress every number_annotations / 100
                processed_annotations += 1
                if number_annotations > 100:
                    if processed_annotations % int(number_annotations / 100) == 0:
                        sendProgress(processed_annotations / number_annotations, 'Computing point intensities',
                                     f"Processing annotation {processed_annotations}/{number_annotations}")
                else:
                    sendProgress(processed_annotations / number_annotations, 'Computing point intensities',
                                 f"Processing annotation {processed_annotations}/{number_annotations}")

    dataset_property_value_dict = {datasetId: property_value_dict}

    sendProgress(0.5, 'Done computing',
                 'Sending computed metrics to the server')
    workerClient.add_multiple_annotation_property_values(
        dataset_property_value_dict)


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
