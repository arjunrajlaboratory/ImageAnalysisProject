import argparse
import json
import sys

import annotation_client.workers as workers

import numpy as np
from skimage import draw


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Channel': {
            'type': 'channel'
        },
        'Radius': {
            'type': 'number',
            'min': 2,
            'max': 10,
            'default': 3,
        },
        'Radius2': {
            'type': 'number',
            'min': 0.5,
            'max': 10,
            'default': 1,
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
    
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationList = workerClient.get_annotation_list_by_shape('point', limit=0)

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    for annotation in annotationList:

        image = workerClient.get_image_for_annotation(annotation)

        if image is None:
            continue

        geojsPoint = annotation['coordinates'][0]
        point = np.array([geojsPoint['y']-0.5, geojsPoint['x']-0.5])
        # Subtract 0.5 to convert from pixel corner to pixel center for skimage.draw.disk

        rr, cc = draw.disk(point, radius, shape=image.shape)
        # Code below seems very inefficient. Probably could just go straight from rr,cc to the calculation. But whatever.
        if rr and cc: # If the circle catches at least one pixel
            mask = np.zeros(image.shape, dtype=bool)
            mask[rr, cc] = 1
            intensity = np.mean(image[mask])

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
