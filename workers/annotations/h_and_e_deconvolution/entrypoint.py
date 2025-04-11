import base64
import argparse
import json
import sys
import pprint

from operator import itemgetter

import annotation_client.tiles as tiles
import annotation_client.workers as workers

from annotation_client.utils import sendProgress, sendError

import numpy as np

from worker_client import WorkerClient

from functools import partial
from skimage import feature, filters, measure, restoration
from skimage.exposure import match_histograms

from skimage import data
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity
import large_image as li


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Max percentile': {
            'type': 'number',
            'min': 0,
            'max': 100,
            'default': 99,
            'tooltip': 'Enter the maximum percentile to rescale the image to.',
            'displayOrder': 1,
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

    tileInfo = tileClient.tiles

    print("tileInfo", tileInfo)

    workerInterface = params['workerInterface']
    maxPercentile = workerInterface['Max percentile']
    if maxPercentile == "":
        maxPercentile = 99

    # If there is an 'IndexRange' key in the tileClient.tiles, then let's get a default value for the batch_xy, batch_z, and batch_time
    if 'IndexRange' in tileInfo:
        if 'IndexXY' in tileInfo['IndexRange']:
            range_xy = range(0, tileInfo['IndexRange']['IndexXY'])
        else:
            range_xy = [0]
        if 'IndexZ' in tileInfo['IndexRange']:
            range_z = range(0, tileInfo['IndexRange']['IndexZ'])
        else:
            range_z = [0]
        if 'IndexT' in tileInfo['IndexRange']:
            range_time = range(0, tileInfo['IndexRange']['IndexT'])
        else:
            range_time = [0]
        if 'IndexC' in tileInfo['IndexRange']:
            range_c = range(0, tileInfo['IndexRange']['IndexC'])
        else:
            range_c = [0]

    else:
        # If there is no 'IndexRange' key in the tileClient.tiles, then there is just one frame
        range_xy = [0]
        range_z = [0]
        range_time = [0]
        range_c = [0]

    print("range_xy", range_xy)
    print("range_z", range_z)
    print("range_time", range_time)
    print("range_c", range_c)

    if max(range_c) != 2:
        sendError("Need 3 channel RGB image")
        return

    gc = tileClient.client

    sink = li.new()

    position_count = 0
    total_positions = len(tileClient.tiles['frames'])

    if 'frames' in tileClient.tiles:

        # Get all position indices
        position_keys = []
        if any('IndexXY' in frame for frame in tileClient.tiles['frames']):
            position_keys.append('IndexXY')
        if any('IndexZ' in frame for frame in tileClient.tiles['frames']):
            position_keys.append('IndexZ')
        if any('IndexT' in frame for frame in tileClient.tiles['frames']):
            position_keys.append('IndexT')

        # Generate unique position tuples
        unique_positions = set(tuple(frame.get(key, 0) for key in position_keys)
                               for frame in tileClient.tiles['frames'])

        # Iterate through positions
        for position in sorted(unique_positions):
            # Create a dict for easier comparison
            position_dict = {key: value for key, value in zip(position_keys, position)}

            # Get all frames for this position
            position_frames = [
                frame for frame in tileClient.tiles['frames']
                if all(frame.get(key, 0) == position_dict.get(key, 0) for key in position_keys)
            ]

            # Determine number of channels
            num_channels = len(position_frames)

            # Load first image to get dimensions
            first_frame = position_frames[0]
            first_frame_index = tileClient.tiles['frames'].index(first_frame)
            first_image = tileClient.getRegion(datasetId, frame=first_frame_index).squeeze()

            # Now initialize im_hed with the right dimensions
            height, width = first_image.shape
            im_rgb = np.zeros((height, width, num_channels), dtype=first_image.dtype)

            # Store first image in the correct channel position
            im_rgb[:, :, first_frame['IndexC']] = first_image

            # Load remaining channel images
            for frame in position_frames[1:]:
                frame_index = tileClient.tiles['frames'].index(frame)
                image = tileClient.getRegion(datasetId, frame=frame_index).squeeze()
                im_rgb[:, :, frame['IndexC']] = image

            # Now im_hed has all channels for this position
            im_hed = rgb2hed(im_rgb)

            # After transformations, write each channel back
            for frame in position_frames:
                # Create parameters for sink.addTile
                large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items()
                                      if k.startswith('Index') and len(k) > 5}

                # Extract the transformed channel
                transformed_channel = rescale_intensity(im_hed[:, :, frame['IndexC']],
                                                        out_range=(0, 255),
                                                        in_range=(0, np.percentile(im_hed[:, :, frame['IndexC']], maxPercentile))).astype(np.uint8)

                # Save the transformed channel
                sink.addTile(transformed_channel, 0, 0, **large_image_params)

            # Update progress
            position_count += 1
            sendProgress(position_count / total_positions, 'Deconvolving',
                         f"Processing position {position_count}/{total_positions}")
    else:
        sendError("Only one image; exiting")
        raise ValueError("Only one image; exiting")

    # Copy over the metadata
    if 'channels' in tileClient.tiles:
        sink.channelNames = tileClient.tiles['channels']

    sink.mm_x = tileClient.tiles['mm_x']
    sink.mm_y = tileClient.tiles['mm_y']
    sink.magnification = tileClient.tiles['magnification']
    sink.write('/tmp/deconvolved.tiff')
    print("Wrote to file")

    item = gc.uploadFileToFolder(datasetId, '/tmp/deconvolved.tiff')
    gc.addMetadataToItem(item['itemId'], {
        'tool': 'H&E Deconvolution',
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
