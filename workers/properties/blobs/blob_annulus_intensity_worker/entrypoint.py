import argparse
import json
import sys
import timeit


import annotation_client.workers as workers
from annotation_client.utils import sendProgress
import annotation_client.tiles as tiles

import numpy as np
from skimage import draw
from skimage import morphology
from collections import defaultdict



def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Channel': {
            'type': 'channel',
        },
        'Radius': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10,
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
    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    
    channel = params['workerInterface']['Channel']
    datasetClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)
    annulus_radius = float(params['workerInterface']['Radius'])

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

    

    for location_key, annotations in grouped_annotations.items():
        time, z, xy = location_key
        frame = datasetClient.coordinatesToFrameIndex(xy, z, time, channel)
        image = datasetClient.getRegion(datasetId, frame=frame)
        image = image.squeeze()

        if image is None:
            continue

        # Compute properties for all annotations at that location
        for annotation in annotations:
            polygon = np.array([list(coordinate.values())[1::-1] for coordinate in annotation['coordinates']])
            mask = draw.polygon2mask(image.shape, polygon)

            # Generate annulus
            selem = morphology.disk(annulus_radius)
            dilated_mask = morphology.binary_dilation(mask, selem)
            annulus_mask = dilated_mask & ~mask  # Subtracting the original mask from dilated mask

            intensities = image[annulus_mask]


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

            processed_annotations += 1
            sendProgress(processed_annotations / number_annotations, 'Computing blob intensity', f"Processing annotation {processed_annotations}/{number_annotations}")
            workerClient.add_annotation_property_values(annotation, prop)
    
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
