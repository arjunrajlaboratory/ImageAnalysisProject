import base64
import argparse
import json
import sys
import pprint

from operator import itemgetter

import annotation_client.tiles as tiles
import annotation_client.workers as workers

from annotation_client.utils import sendProgress, sendError

import imageio
import numpy as np

from worker_client import WorkerClient

from functools import partial
from skimage import feature, filters, measure, restoration
from skimage.exposure import match_histograms

import large_image as li


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Reference XY Coordinate': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 8',
                'label': 'Reference XY coordinate.',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the XY coordinate of the reference image to match the histogram of.'
            },
            'displayOrder': 1
        },
        'Reference Z Coordinate': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 8',
                'label': 'Reference Z coordinate.',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Z coordinate of the reference image to match the histogram of.'
            },
            'displayOrder': 2
        },
        'Reference Time Coordinate': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 8',
                'label': 'Reference Time coordinate.',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Time positions to retain. Separate multiple groups with a comma.'
            },
            'displayOrder': 3
        },
        'Channels to correct': {
            'type': 'channelCheckboxes',
            'tooltip': 'Process selected channels.',
            'displayOrder': 2,
        },
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

    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    workerInterface = params['workerInterface']
    if workerInterface['Reference XY Coordinate'] == "":
        reference_XY = 0
    else:
        reference_XY = int(workerInterface['Reference XY Coordinate']) - 1
    if workerInterface['Reference Z Coordinate'] == "":
        reference_Z = 0
    else:
        reference_Z = int(workerInterface['Reference Z Coordinate']) - 1
    if workerInterface['Reference Time Coordinate'] == "":
        reference_Time = 0
    else:
        reference_Time = int(workerInterface['Reference Time Coordinate']) - 1
    allChannels = workerInterface['Channels to correct']

    print("allChannels", allChannels)
    # Output is allChannels {'1': True, '2': True}
    # This means that channels 1 and 2 are being blurred
    channels = [int(k) for k, v in allChannels.items() if v]
    print("channels", channels)
    if len(channels) == 0:
        sendError("No channels to correct")
        return

    # Get reference images for each channel
    reference_images = {}
    for channel in channels:
        frame = tileClient.coordinatesToFrameIndex(
            reference_XY, reference_Z, reference_Time, channel)
        reference_images[channel] = tileClient.getRegion(
            datasetId, frame=frame).squeeze()

    gc = tileClient.client

    sink = li.new()

    if 'frames' in tileClient.tiles:
        for i, frame in enumerate(tileClient.tiles['frames']):
            # Create a parameters dictionary with only the indices that exist in frame
            # The len(k) > 5 is to avoid the 'Index' key that has no postfix to it
            large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items(
            ) if k.startswith('Index') and len(k) > 5}

            image = tileClient.getRegion(datasetId, frame=i).squeeze()
            if frame['IndexC'] in channels:
                # Only process the channel that is being processed
                image = match_histograms(
                    image, reference_images[frame['IndexC']])

            sink.addTile(image, 0, 0, **large_image_params)

            sendProgress(i / len(tileClient.tiles['frames']), 'Histogram matching',
                         f"Processing frame {i+1}/{len(tileClient.tiles['frames'])}")
    else:
        sendError("Only one image; exiting")
        return

    # Copy over the metadata
    if 'channels' in tileClient.tiles:
        sink.channelNames = tileClient.tiles['channels']

    sink.mm_x = tileClient.tiles['mm_x']
    sink.mm_y = tileClient.tiles['mm_y']
    sink.magnification = tileClient.tiles['magnification']
    sink.write('/tmp/normalized.tiff')
    print("Wrote to file")

    item = gc.uploadFileToFolder(datasetId, '/tmp/normalized.tiff')
    gc.addMetadataToItem(item['itemId'], {
        'tool': 'Histogram matching',
        'reference_XY': reference_XY,
        'reference_Z': reference_Z,
        'reference_Time': reference_Time,
    })
    print("Uploaded file")


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='Compute average intensity values in a circle around point annotations')

    parser.add_argument('--datasetId', type=str,
                        required=False, action='store')
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
