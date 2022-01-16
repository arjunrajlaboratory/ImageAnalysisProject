import argparse
import json
import sys

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles

import imageio
import numpy as np
from skimage import filters, measure


def main(datasetId, apiUrl, token, params):
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
    propertyName = params.get('customName', None)
    if not propertyName:
        propertyName = params.get('name', 'unknown_property')

    annotationIds = params.get('annotationIds', None)

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    annotationList = []
    if annotationIds:
        # Get the annotations specified by id in the parameters
        for id in annotationIds:
            annotationList.append(annotationClient.getAnnotationById(id))
    else:
        # Get all point annotations from the dataset
        annotationList = annotationClient.getAnnotationsByDatasetId(
            datasetId, shape='point')

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    # Constants
    block_size = 25

    # Cache downloaded images by location
    images = {}

    for annotation in annotationList:
        # Get image location
        channel = annotation['channel']
        location = annotation['location']
        time, z, xy = location['Time'], location['Z'], location['XY']

        # Look for cached image. Initialize cache if necessary.
        image = images.setdefault(channel, {}).setdefault(
            time, {}).setdefault(z, {}).get(xy, None)

        if image is None:
            # Download the image at specified location
            pngBuffer = datasetClient.getRawImage(xy, z, time, channel)

            # Read the png buffer
            image = imageio.imread(pngBuffer)

            # Cache the image
            images[channel][time][z][xy] = image

        geojsPoint = annotation['coordinates'][0]
        point = np.array([round(geojsPoint['y']), round(geojsPoint['x'])])

        li = point - int((block_size - 1) / 2)
        ui = li + block_size
        i = get_indices([*li, *ui], image)
        center = point - i[:2]

        crop = image[i[0]:i[2], i[1]:i[3]]
        binary = crop > filters.threshold_otsu(crop)
        binary_labeled = measure.label(binary)
        cell_binary = (binary_labeled == binary_labeled[center[0], center[1]])
        cell = crop[cell_binary]
        intensity = np.mean(cell)

        annotationClient.addAnnotationPropertyValues(datasetId, annotation['_id'], {
            propertyName: float(intensity)})


def get_indices(i, image):
    i[0] = max(i[0], 0)
    i[1] = max(i[1], 0)
    i[2] = min(i[2], image.shape[0])
    i[3] = min(i[3], image.shape[1])

    return i


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in a circle around point annotations')

    parser.add_argument('--datasetId', type=str, required=True, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str,
                        required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    main(args.datasetId, args.apiUrl, args.token, json.loads(args.parameters))
