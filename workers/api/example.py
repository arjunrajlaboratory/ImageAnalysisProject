import argparse
import json
import sys

from worker_api import WorkerClient

import numpy as np


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

    workerClient = WorkerClient(datasetId, apiUrl, token, params)
    annotationList = workerClient.get_annotation_list('point')

    # We need at least one annotation
    if len(annotationList) == 0:
        return

    for annotation in annotationList:

        image = workerClient.get_image_for_annotation(annotation)

        if image is None:
            continue

        geojsPoint = annotation['coordinates'][0]
        point = np.array([round(geojsPoint['y']), round(geojsPoint['x'])])

        intensity = image[point[0], point[1]]

        workerClient.add_annotation_property_values(annotation, int(intensity))


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
