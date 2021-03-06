import argparse
import json
import sys

import annotation_client.workers as workers

import numpy as np
from skimage import draw


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

    # Constants
    radius = 5

    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    annotationList = workerClient.get_annotation_list_by_shape('point')

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    for annotation in annotationList:

        image = workerClient.get_image_for_annotation(annotation)

        if image is None:
            continue

        geojsPoint = annotation['coordinates'][0]
        point = np.array([round(geojsPoint['y']), round(geojsPoint['x'])])

        rr, cc = draw.disk(point, radius, shape=image.shape)
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
    parser.add_argument('--parameters', type=str,
                        required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    main(args.datasetId, args.apiUrl, args.token, json.loads(args.parameters))
