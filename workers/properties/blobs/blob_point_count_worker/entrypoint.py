import argparse
import json
import sys

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles

import imageio
import numpy as np
from point_in_polygon import point_in_polygon

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
        pointList = annotationClient.getAnnotationsByDatasetId(
            datasetId, shape='point')
        annotationList = annotationClient.getAnnotationsByDatasetId(
            datasetId, shape='polygon')

    # We need at least one annotation
    if len(annotationList) == 0:
        return

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

        polygon = np.array([list(coordinate.values())[1::-1] for coordinate in annotation['coordinates']])
        point_count = np.sum(point_in_polygon(points,polygon))

        annotationClient.addAnnotationPropertyValues(datasetId, annotation['_id'], {
            propertyName: int(point_count)})


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
