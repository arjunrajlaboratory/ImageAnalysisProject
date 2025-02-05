import base64
import argparse
import json
import sys
import pprint

from operator import itemgetter

import annotation_client.tiles as tiles
import annotation_client.workers as workers
import annotation_client.annotations as annotations

from annotation_client.utils import sendProgress, sendError

import annotation_utilities.annotation_tools as annotation_tools
import annotation_utilities.batch_argument_parser as batch_argument_parser

import imageio
import numpy as np

from worker_client import WorkerClient

from functools import partial
from skimage import feature, filters, measure, restoration
from pystackreg import StackReg

import large_image as li


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'Apply to XY coordinates': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-7; default is all',
                'label': 'Apply to XY coordinates.',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the XY coordinates to apply the histogram matching to. Separate multiple groups with a comma. Default is all.'
            },
            'displayOrder': 1
        },
        'Reference Z Coordinate': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 8; default is 0',
                'label': 'Reference Z coordinate.',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Z coordinate to reference for registration. Default is 0.'
            },
            'displayOrder': 2
        },
        'Reference Time Coordinate': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 8; default is 0',
                'label': 'Reference Time coordinate.',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the time point to reference for registration. Default is 0.'
            },
            'displayOrder': 3
        },
        'Reference Channel': {
            'type': 'channel',
            'tooltip': 'Select the channel to reference for registration.',
            'displayOrder': 4
        },
        'Channels to correct': {
            'type': 'channelCheckboxes',
            'tooltip': 'Process selected channels.',
            'displayOrder': 5,
        },
        'Reference region tag': {
            'type': 'tags',
            'tooltip': 'Enter the tag of the region to reference for registration.',
            'displayOrder': 6
        },
        'Algorithm': {
            'type': 'select',
            'options': ['Translation', 'Rigid', 'Affine'],
            'label': 'Algorithm',
            'default': 'Translation',
            'displayOrder': 7
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

    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)

    # Get algorithm from workerInterface
    algorithm = params['workerInterface']['Algorithm']

    tileInfo = tileClient.tiles

    print("tileInfo", tileInfo)

    # First check whether we have multiple images
    # Then check that the time dimension exists
    if 'IndexRange' not in tileInfo:
        sendError("Just one image; exiting")
        return
    else:
        if 'IndexT' not in tileInfo['IndexRange']:
            sendError("Time dimension not found; exiting")
            return

    if 'IndexXY' in tileInfo['IndexRange']:
        range_xy = range(0, tileInfo['IndexRange']['IndexXY'])
    else:
        range_xy = [0]

    # range_t = range(0, tileInfo['IndexRange']['IndexT'])

    workerInterface = params['workerInterface']
    if workerInterface['Apply to XY coordinates'] == "":
        apply_XY = range_xy
    else:
        apply_XY = batch_argument_parser.process_range_list(
            workerInterface['Apply to XY coordinates'], convert_one_to_zero_index=True)
        apply_XY = list(set(apply_XY) & set(range_xy))
    # TODO: Could add guards to make sure that the reference Z and Time are within the range of the dataset
    if workerInterface['Reference Z Coordinate'] == "":
        reference_Z = 0
    else:
        reference_Z = int(workerInterface['Reference Z Coordinate']) - 1
    if workerInterface['Reference Time Coordinate'] == "":
        reference_Time = 0
    else:
        reference_Time = int(workerInterface['Reference Time Coordinate']) - 1

    reference_channel = workerInterface['Reference Channel']
    if reference_channel == "" or reference_channel == -1:
        reference_channel = 0

    allChannels = workerInterface['Channels to correct']

    print("allChannels", allChannels)
    # Output is allChannels {'1': True, '2': True}
    # This means that channels 1 and 2 are being blurred
    channels = [int(k) for k, v in allChannels.items() if v]
    print("channels", channels)
    if len(channels) == 0:
        sendError("No channels to correct")
        return

    # Okay, now let's get the crop rectangle (could also be a blob)
    should_use_reference_region = params['workerInterface'][
        'Reference region tag'] is not None and params['workerInterface']['Reference region tag']
    if should_use_reference_region:
        blobAnnotationList = annotationClient.getAnnotationsByDatasetId(
            datasetId, limit=1000, shape='polygon')
        rectangleAnnotationList = annotationClient.getAnnotationsByDatasetId(
            datasetId, limit=1000, shape='rectangle')
        # Add the rectangle annotations to the blob annotations
        blobAnnotationList.extend(rectangleAnnotationList)

        cropAnnotationList = annotation_tools.get_annotations_with_tags(
            blobAnnotationList, params['workerInterface']['Crop Rectangle'], exclusive=False)
        if cropAnnotationList is None or len(cropAnnotationList) == 0:
            sendError("No crop rectangle found")
            return
        else:
            cropAnnotation = cropAnnotationList[0]

            # Extract x and y coordinates
            x_coords = [coord['x'] for coord in cropAnnotation['coordinates']]
            y_coords = [coord['y'] for coord in cropAnnotation['coordinates']]

            # Calculate bounding box
            reference_region_left = min(x_coords)
            reference_region_top = min(y_coords)
            reference_region_right = max(x_coords)
            reference_region_bottom = max(y_coords)

            print(
                f"Reference region dimensions - left: {reference_region_left}, top: {reference_region_top}, right: {reference_region_right}, bottom: {reference_region_bottom}")

    # Initialize the stackreg object
    # TODO: Make this configurable based on selected algorithm
    sr = StackReg(StackReg.TRANSLATION)

    # Now let's compute the registration matrices
    registration_matrices = {}
    progress_counter = 0
    total_progress = len(apply_XY) * tileInfo['IndexRange']['IndexT']
    for xy in apply_XY:
        # Set first matrix to identity
        registration_matrices[(xy, 0)] = np.eye(3)

        # Get first frame
        frame = tileClient.coordinatesToFrameIndex(
            xy, reference_Z, 0, reference_channel)
        # TODO: Can use the reference_region to pull just the relevant region, look at the crop tool for an example
        current_image = tileClient.getRegion(datasetId, frame=frame).squeeze()

        for t in range(1, tileInfo['IndexRange']['IndexT']):
            # Get the reference image
            next_frame = tileClient.coordinatesToFrameIndex(
                xy, reference_Z, t, reference_channel)
            next_image = tileClient.getRegion(
                datasetId, frame=next_frame).squeeze()

            # Compute the registration matrix
            registration_matrix = sr.register(current_image, next_image)
            cumulative_registration_matrix = np.dot(
                registration_matrix, registration_matrices[(xy, t-1)])
            registration_matrices[(xy, t)] = cumulative_registration_matrix
            current_image = next_image

            sendProgress(progress_counter / total_progress,
                         'Calculating registration matrices',
                         f"Processing t: {t}, xy: {xy}")
            progress_counter += 1

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
                # First check if frame even has a "IndexXY" key
                if 'IndexXY' in frame:
                    if frame['IndexXY'] in apply_XY:
                        transformed_image = sr.transform(
                            image, tmat=registration_matrices[(frame['IndexXY'], frame['IndexT'])])
                        image = transformed_image.astype(image.dtype)
                else:
                    transformed_image = sr.transform(
                        image, tmat=registration_matrices[(0, frame['IndexT'])])
                    image = transformed_image.astype(image.dtype)

            sink.addTile(image, 0, 0, **large_image_params)

            sendProgress(i / len(tileClient.tiles['frames']), 'Registration',
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
    sink.write('/tmp/registered.tiff')
    print("Wrote to file")

    item = gc.uploadFileToFolder(datasetId, '/tmp/registered.tiff')
    gc.addMetadataToItem(item['itemId'], {
        'tool': 'Registration',
        'apply_XY': apply_XY,
        'reference_Z': reference_Z,
        'reference_Time': reference_Time,
        'reference_channel': reference_channel,
        'algorithm': algorithm,
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
        # case 'preview':
        #     preview(datasetId, apiUrl, token, params, params['image'])
