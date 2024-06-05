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
from collections import defaultdict

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Channel 1': {
            'type': 'channel'
        },
        'Channel 2': {
            'type': 'channel'
        },
        'Threshold percentile': {
            'type': 'number',
            'default': 50,
            'min': 0,
            'max': 100,
            'step': 0.1
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
    channel1 = params['workerInterface']['Channel 1']
    channel2 = params['workerInterface']['Channel 2']
    threshold_percentile = params['workerInterface']['Threshold percentile']
    datasetClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    # Following line should be updated to get just the annotations with specified tags
    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, params.get('tags', {}).get('tags', []), params.get('tags', {}).get('exclusive', False))

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    start_time = timeit.default_timer()

    grouped_annotations = defaultdict(list)
    for annotation in annotationList:
        location_key = (annotation['location']['Time'], annotation['location']['Z'], annotation['location']['XY'])
        grouped_annotations[location_key].append(annotation)
    
    number_annotations = len(annotationList)
    
    # For reporting progress
    processed_annotations = 0

    property_value_dict = {}  # Initialize as a dictionary

    for location_key, annotations in grouped_annotations.items():
        time, z, xy = location_key
        frame1 = datasetClient.coordinatesToFrameIndex(xy, z, time, channel1)
        image1 = datasetClient.getRegion(datasetId, frame=frame1)
        frame2 = datasetClient.coordinatesToFrameIndex(xy, z, time, channel2)
        image2 = datasetClient.getRegion(datasetId, frame=frame2)

        if image1 is None or image2 is None:
            continue

        # Compute properties for all annotations at that location
        for annotation in annotations:
            polygon = np.array([list(coordinate.values())[1::-1] for coordinate in annotation['coordinates']])
            mask = draw.polygon2mask(image1.shape, polygon) # Using the shape of the first channel image, could be either channel
            # First, let's get the intensities of the two channels
            intensities1 = image1[mask]
            intensities2 = image2[mask]

            # Now, let's calculate the percentile intensity of the two channels and create a unified mask for both channels
            percentile_intensity1 = np.percentile(intensities1, threshold_percentile)
            percentile_intensity2 = np.percentile(intensities2, threshold_percentile)
            
            # We will use the median intensity of the two channels to create a mask for both channels
            mask1 = (intensities1 > percentile_intensity1)
            mask2 = (intensities2 > percentile_intensity2)
            combined_mask  = np.logical_or(mask1, mask2)

            # Now let's get the mean intensities of the two channels using the combined mask
            mean_intensity1 = np.mean(intensities1[combined_mask])
            mean_intensity2 = np.mean(intensities2[combined_mask])
            
            median_intensity1 = np.median(intensities1)
            q25_intensity1 = np.percentile(intensities1, 25)
            q40_intensity1 = np.percentile(intensities1, 40)
            q75_intensity1 = np.percentile(intensities1, 75)

            median_intensity2 = np.median(intensities2)
            q25_intensity2 = np.percentile(intensities2, 25)
            q40_intensity2 = np.percentile(intensities2, 40)
            q75_intensity2 = np.percentile(intensities2, 75)

            prop = {
                'Channel 1': {
                    'MeanColonyIntensity': float(mean_intensity1),
                    'MedianIntensity': float(median_intensity1),
                    '25thPercentileIntensity': float(q25_intensity1),
                    '40thPercentileIntensity': float(q40_intensity1),
                    '75thPercentileIntensity': float(q75_intensity1),
                },
                'Channel 2': {
                    'MeanColonyIntensity': float(mean_intensity2),
                    'MedianIntensity': float(median_intensity2),
                    '25thPercentileIntensity': float(q25_intensity2),
                    '40thPercentileIntensity': float(q40_intensity2),
                    '75thPercentileIntensity': float(q75_intensity2),
                }
            }

            property_value_dict[annotation['_id']] = prop
            processed_annotations += 1
            sendProgress(processed_annotations / number_annotations, 'Computing colony intensity', f"Processing annotation {processed_annotations}/{number_annotations}")
    
    dataset_property_value_dict = {datasetId: property_value_dict}

    sendProgress(0.5,'Done computing', 'Sending computed metrics to the server')
    workerClient.add_multiple_annotation_property_values(dataset_property_value_dict)
    
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Executed the code in: {execution_time} seconds")


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
