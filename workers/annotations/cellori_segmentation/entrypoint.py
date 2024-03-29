import argparse
import json
import sys

from itertools import product
from operator import itemgetter

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles
import annotation_client.workers as workers

import numpy as np  # library for array manipulation
from cellori import CelloriSegmentation
from rasterio.features import shapes

import utils


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Nuclei Channel': {
            'type': 'channel',
            'default': -1,
            'required': False
        },
        'Cytoplasm Channel': {
            'type': 'channel',
            'default': -1,
            'required': False
        },
        'Diameter': {
            'type': 'number',
            'min': 0,
            'max': 200,
            'default': 10
        },
        'Batch XY': {
            'type': 'text'
        },
        'Batch Z': {
            'type': 'text'
        },
        'Batch Time': {
            'type': 'text'
        }
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
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
    keys = ["assignment", "channel", "connectTo", "tags", "tile", "workerInterface"]
    if not all(key in params for key in keys):
        print ("Invalid worker parameters", params)
        return
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(*keys)(params)

    # Get the model and diameter from interface values
    nuclei_channel = workerInterface.get('Nuclei Channel', None)
    cytoplasm_channel = workerInterface.get('Cytoplasm Channel', None)
    diameter = float(workerInterface['Diameter'])
    batch_xy = workerInterface.get('Batch XY', None)
    batch_z = workerInterface.get('Batch Z', None)
    batch_time = workerInterface.get('Batch Time', None)

    batch_xy = utils.process_range_list(batch_xy)
    batch_z = utils.process_range_list(batch_z)
    batch_time = utils.process_range_list(batch_time)

    if batch_xy is None:
        batch_xy = [tile['XY'] + 1]
    if batch_z is None:
        batch_z = [tile['Z'] + 1]
    if batch_time is None:
        batch_time = [tile['Time'] + 1]

    # Setup helper classes with url and credentials
    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)
    datasetClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    for xy, z, time in product(batch_xy, batch_z, batch_time):

        xy -= 1
        z -= 1
        time -= 1

        # TODO: will need to iterate or stitch and handle roi and proper intensities

        if (nuclei_channel is not None) and (nuclei_channel > -1):
            nuclei_frame = datasetClient.coordinatesToFrameIndex(xy, z, time, nuclei_channel)
            nuclei_image = datasetClient.getRegion(datasetId, frame=nuclei_frame).squeeze()
        else:
            nuclei_image = None
        if (cytoplasm_channel is not None) and (cytoplasm_channel > -1):
            cytoplasm_frame = datasetClient.coordinatesToFrameIndex(xy, z, time, cytoplasm_channel)
            cytoplasm_image = datasetClient.getRegion(datasetId, frame=cytoplasm_frame).squeeze()
        else:
            cytoplasm_image = None

        image = None
        if (cytoplasm_image is not None) & (nuclei_image is not None):
            image = np.stack((cytoplasm_image, nuclei_image))
        elif cytoplasm_image is not None:
            image = cytoplasm_image

        model = CelloriSegmentation(model='cyto', batch_size=1)

        masks, _ = model.predict(image, diameter=diameter)
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
                    "XY": xy,
                    "Z": z,
                    "Time": time
                },
                "datasetId": datasetId,
                "coordinates": [{"x": float(x), "y": float(y), "z": 0} for x, y in polygon['coordinates'][0]]
            }
            annotationClient.createAnnotation(annotation)
            # if count > 1000:  # TODO: arbitrary limit to avoid flooding the server if threshold is too big
            #     break
            count = count + 1


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
