import argparse
import json
import sys

import annotation_client.workers as workers

import numpy as np
from skimage import filters, measure


def get_indices(i, image):
    i[0] = max(i[0], 0)
    i[1] = max(i[1], 0)
    i[2] = min(i[2], image.shape[0])
    i[3] = min(i[3], image.shape[1])

    return i


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Channel': {
            'type': 'channel'
        }
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
    annotationList = workerClient.get_annotation_list_by_shape('point', limit=0)

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    # Constants
    block_size = 25

    for annotation in annotationList:

        image = workerClient.get_image_for_annotation(annotation)

        if image is None:
            continue

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

        workerClient.add_annotation_property_values(annotation, float(intensity))


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
