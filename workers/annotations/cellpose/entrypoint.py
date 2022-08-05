import argparse
import json
import sys

from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers

import numpy as np  # library for array manipulation
from cellpose import models
from rasterio.features import shapes


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Model': {
            'type': 'text',
            'default': 'cyto'
        },
        'Diameter': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


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
    # Check whether we need to preview, send the interface, or compute
    request = params.get('request', 'compute')
    if request == 'interface':
        return interface(params['image'], apiUrl, token)

    # roughly validate params
    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    if not all(key in params for key in keys):
        print ("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)

    # Get the model and diameter from interface values
    model = workerInterface['Model']['value']
    diameter = float(workerInterface['Diameter']['value'])

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    # TODO: will need to iterate or stitch and handle roi and proper intensities
    frame = datasetClient.coordinatesToFrameIndex(tile['XY'], tile['Z'], tile['Time'], channel)
    image = datasetClient.getRegion(datasetId, frame=frame).squeeze()

    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(model_type=model)

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    channels = [[0, 0]]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended)
    # diameter can be a list or a single number for all images

    masks, _, _, _ = model.eval(image, diameter=diameter, channels=channels)
    polygons = shapes(masks.astype(np.int32), masks > 0)

    # Upload annotations TODO: handle connectTo. could be done server-side via special api flag ?
    print(f"Uploading {masks.max()} annotations")
    count = 0
    for polygon, _ in polygons:
        annotation = {
            "tags": tags,
            "shape": "polygon",
            "channel": channel,
            "location": {
                "XY": assignment['XY'],
                "Z": assignment['Z'],
                "Time": assignment['Time']
            },
            "datasetId": datasetId,
            "coordinates": [{"x": float(x), "y": float(y), "z": 0} for x, y in polygon['coordinates'][0]]
        }
        annotationClient.createAnnotation(annotation)
        if count > 1000:  # TODO: arbitrary limit to avoid flooding the server if threshold is too big
            break
        count = count + 1


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
