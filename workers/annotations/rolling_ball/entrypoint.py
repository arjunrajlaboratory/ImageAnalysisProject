import base64
import argparse
import json
import sys

from operator import itemgetter

import annotation_client.tiles as tiles
import annotation_client.workers as workers

from annotation_client.utils import sendProgress, sendError, sendWarning

import annotation_utilities.annotation_tools as annotation_tools

import imageio
import numpy as np


from skimage import filters, restoration


def preview(datasetId, apiUrl, token, params, bimage):
    # Setup helper classes with url and credentials
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)
    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    keys = ["assignment", "channel", "connectTo",
            "tags", "tile", "workerInterface"]
    assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(
        *keys)(params)
    sigma = float(workerInterface['Sigma'])

    # Get the tile
    frame = tileClient.coordinatesToFrameIndex(
        tile['XY'], tile['Z'], tile['Time'], channel)
    image = tileClient.getRegion(datasetId, frame=frame).squeeze()

    (width, height) = np.shape(image)

    blurred = filters.gaussian(image, sigma=sigma)*255
    blurred = blurred.astype(np.uint8)

    # Convert image to RGB
    rgba = np.zeros((width, height, 4), np.uint8)

    rgba[:, :, 0] = blurred
    rgba[:, :, 1] = blurred
    rgba[:, :, 2] = blurred
    rgba[:, :, 3] = 255

    # Generate an output data-uri from the threshold image
    outputPng = imageio.imwrite('<bytes>', rgba, format='png')
    data64 = base64.b64encode(outputPng)
    dataUri = 'data:image/png;base64,' + data64.decode('ascii')

    # Send the preview object to the server
    preview = {
        'image': dataUri
    }
    client.setWorkerImagePreview(bimage, preview)


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Radius': {
            'type': 'number',
            'min': 0,
            'max': 100,
            'default': 20,
            'tooltip': 'The radius of the rolling ball.',
            'displayOrder': 0,
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

    # Lazy import: keeps large_image off the interface/preview path; only needed during compute. See todo/worker-startup-latency.md
    import large_image as li

    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    workerInterface = params['workerInterface']
    radius = float(workerInterface['Radius'])
    allChannels = workerInterface.get('Channels to correct')

    print("allChannels", allChannels)
    # The front end sends {'1': True, '2': True}, meaning channels 1 and 2 are
    # being corrected. Any other shape is malformed and is rejected below.
    try:
        channels = annotation_tools.get_selected_channels(
            allChannels, 'Channels to correct')
    except ValueError as exc:
        sendError("Could not read the channel selection.", info=str(exc))
        return
    # An empty selection is not an error here: the worker still writes out a
    # copy of the dataset with no channel corrected.
    #
    # A selection that names channels the dataset does not have is different: the
    # per-frame filter below would match nothing and the worker would upload an
    # untouched copy of the input while reporting success. A single-channel dataset
    # run with a config that selects channel 1 does exactly that.
    num_channels = tileClient.tiles.get('IndexRange', {}).get('IndexC', 1)
    channels, missing_channels = annotation_tools.split_channel_selection(
        channels, num_channels)
    if missing_channels:
        detail = (f"Selected channel indices {missing_channels} do not exist in "
                  f"this dataset, which has {num_channels} channel(s) "
                  f"(indices 0-{num_channels - 1}).")
        if not channels:
            sendError("None of the selected channels exist in this dataset.",
                      info=detail)
            return
        sendWarning("Ignoring channels this dataset does not have.", info=detail)
    print("channels", channels)

    tile = params['tile']
    frame = tileClient.coordinatesToFrameIndex(
        tile['XY'], tile['Z'], tile['Time'], params['channel'])

    gc = tileClient.client

    sink = li.new()

    if 'frames' in tileClient.tiles:
        for i, frame in enumerate(tileClient.tiles['frames']):
            large_image_params = annotation_tools.frame_to_large_image_params(
                frame)

            image = tileClient.getRegion(datasetId, frame=i).squeeze()
            # A single-channel dataset omits IndexC from the frame entirely; channel
            # 0 is then the only channel there is.
            if annotation_tools.get_frame_index(frame, 'IndexC') in channels:
                # Only process the channel that is being processed
                background = restoration.rolling_ball(
                    image, radius=radius)
                image = image-background

            sink.addTile(image, 0, 0, **large_image_params)

            sendProgress(i / len(tileClient.tiles['frames']), 'Rolling ball',
                         f"Processing frame {i+1}/{len(tileClient.tiles['frames'])}")
    else:
        image = tileClient.getRegion(datasetId, frame=frame).squeeze()
        background = restoration.rolling_ball(image, radius=radius)
        image = image-background
        sink.addTile(image, 0, 0, z=0)  # X, Y, Z

    # Copy over the metadata
    if 'channels' in tileClient.tiles:
        sink.channelNames = tileClient.tiles['channels']

    sink.mm_x = tileClient.tiles['mm_x']
    sink.mm_y = tileClient.tiles['mm_y']
    sink.magnification = tileClient.tiles['magnification']
    sink.write('/tmp/output.tiff')
    print("Wrote to file")

    item = gc.uploadFileToFolder(datasetId, '/tmp/output.tiff')
    gc.addMetadataToItem(item['itemId'], {
        'tool': 'Rolling ball',
        'radius': radius,
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
        case 'preview':
            preview(datasetId, apiUrl, token, params, params['image'])
