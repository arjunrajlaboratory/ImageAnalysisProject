import argparse
import json
import sys

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles

import imageio
import numpy as np

from skimage import filters
from skimage.feature import peak_local_max


def main(datasetId, apiUrl, token, params):
    """
    params (could change):
        configurationId,
        datasetId,
        description: tool description,
        type: tool type,
        id: tool id,
        name: tool name,
        image: docker image,
        channel: annotation channel,
        assignment: annotation assignment ({XY, Z, Time}),
        tags: annotation tags (list of strings),
        tile: tile position (TODO: roi) ({XY, Z, Time}),
        connectTo: how new annotations should be connected
    """
    # roughly validate params
    keys = ["assignment", "channel", "connectTo", "tags", "tile"]
    if not all(key in params for key in keys):
        print("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile = itemgetter(*keys)(params)

    # TODO: completely arbitrary
    threshold = 0.001

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    # TODO: will need to iterate or stitch and handle roi and proper intensities
    stack = list()

    for z in range(datasetClient.tiles['IndexRange']['IndexZ']):
        pngBuffer = datasetClient.getRawImage(tile['XY'], z, tile['Time'], channel)
        stack.append(imageio.imread(pngBuffer))

    stack = np.stack(stack)

    # Filter
    gaussian = filters.gaussian(stack, sigma=2, mode='nearest')
    laplacian = filters.laplace(gaussian)

    # Find local maxima
    coordinates = peak_local_max(laplacian, min_distance=3)
    values = laplacian[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]

    # Threshold maxima
    index = values > threshold
    thresholdCoordinates = coordinates[index, :]

    # Upload annotations TODO: handle connectTo. could be done server-side via special api flag ?
    print("Uploading {} annotations".format(len(thresholdCoordinates)))
    count = 0
    for [z, y, x] in thresholdCoordinates:
        annotation = {
            "tags": tags,
            "shape": "point",
            "channel": channel,
            "location": {
                "XY": assignment['XY'],
                "Z": int(z),
                "Time": assignment['Time']
            },
            "datasetId": datasetId,
            "coordinates": [{"x": float(x), "y": float(y), "z": 0}]
        }
        annotationClient.createAnnotation(annotation)
        print("uploading annotation ", z, x, y)
        if count > 1000:  # TODO: arbitrary limit to avoid flooding the server if threshold is too big
            break
        count = count + 1


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
