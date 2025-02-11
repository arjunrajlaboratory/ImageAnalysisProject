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

import imageio
import numpy as np

from worker_client import WorkerClient
import annotation_utilities.batch_argument_parser as batch_argument_parser

from functools import partial
from skimage import feature, filters, measure, restoration

import large_image as li


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    # Available types: number, text, tags, layer
    interface = {
        'XY Range': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the XY positions you want to retain (default is all)',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the XY positions to retain. Separate multiple groups with a comma.'
            },
            'displayOrder': 1
        },
        'Z Range': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the Z positions you want to retain (default is all)',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Z positions to retain. Separate multiple groups with a comma.'
            },
            'displayOrder': 2
        },
        'Time Range': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the Time positions you want to retain (default is all)',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Time positions to retain. Separate multiple groups with a comma.'
            },
            'displayOrder': 3
        },
        # TODO: We can add this back in later. For now, the problem
        # is that the front-end interface can't handle channel definitions
        # changing underneath, so it doesn't work properly.
        # 'Channels to keep': {
        #     'type': 'channelCheckboxes',
        #     'tooltip': 'Process selected channels.',
        #     'displayOrder': 4,
        # },
        'Crop Rectangle': {
            'type': 'tags',
            'tooltip': 'Select tag of the crop rectangle. Will take the first rectangle or blob with the specified tag.',
            'displayOrder': 4,
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

    tileClient = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId)

    annotationClient = annotations.UPennContrastAnnotationClient(
        apiUrl=apiUrl, token=token)

    tileInfo = tileClient.tiles

    print("tileClient.tiles", tileInfo)

    batch_xy = params['workerInterface']['XY Range']
    batch_z = params['workerInterface']['Z Range']
    batch_time = params['workerInterface']['Time Range']

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

    # Check strings and then convert from iterators to lists
    if batch_xy is not None and batch_xy.strip():
        batch_xy = list(batch_argument_parser.process_range_list(
            batch_xy, convert_one_to_zero_index=True))
        batch_xy = [x for x in batch_xy if x in range_xy]
    else:
        batch_xy = range_xy

    if batch_z is not None and batch_z.strip():
        batch_z = list(batch_argument_parser.process_range_list(
            batch_z, convert_one_to_zero_index=True))
        batch_z = [x for x in batch_z if x in range_z]
    else:
        batch_z = range_z

    if batch_time is not None and batch_time.strip():
        batch_time = list(batch_argument_parser.process_range_list(
            batch_time, convert_one_to_zero_index=True))
        batch_time = [x for x in batch_time if x in range_time]
    else:
        batch_time = range_time

    # TODO: Leaving this logic in place for now, but nothing is really being
    # done with it. If we implement channel selection, we can add in the
    # logic to select only certain channels here as per above.
    channels = range_c

    print("batch_xy", batch_xy)
    print("batch_z", batch_z)
    print("batch_time", batch_time)

    # TODO: As noted above, while we are able to keep only selected channels
    # in the output large_image, the front-end is unable to handle this change
    # allChannels = params['workerInterface']['Channels to keep']
    # print("allChannels", allChannels)
    # # Output is allChannels {'1': True, '2': True}
    # # This means that channels 1 and 2 are being blurred
    # channels = [int(k) for k, v in allChannels.items() if v]
    # print("channels", channels)

    # Okay, now let's get the crop rectangle (could also be a blob)
    should_crop_to_rectangle = params['workerInterface'][
        'Crop Rectangle'] is not None and params['workerInterface']['Crop Rectangle']
    if should_crop_to_rectangle:
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
            left = min(x_coords)
            top = min(y_coords)
            right = max(x_coords)
            bottom = max(y_coords)

            print(
                f"Crop dimensions - left: {left}, top: {top}, right: {right}, bottom: {bottom}")

    gc = tileClient.client

    sink = li.new()

    if 'frames' in tileClient.tiles:
        for i, frame in enumerate(tileClient.tiles['frames']):
            # Create a parameters dictionary with only the indices that exist in frame
            # The len(k) > 5 is to avoid the 'Index' key that has no postfix to it
            large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items(
            ) if k.startswith('Index') and len(k) > 5}

            # Check if the frame indices match our batch selections
            should_process = True
            new_params = {}

            # Check each parameter against its corresponding batch list
            param_batch_mapping = {
                'xy': (batch_xy, 'xy'),
                'z': (batch_z, 'z'),
                't': (batch_time, 't'),
                'c': (channels, 'c')
            }

            for batch_list, param_key in param_batch_mapping.values():
                if param_key in large_image_params:
                    value = large_image_params[param_key]
                    if value not in batch_list:
                        should_process = False
                        break
                    new_params[param_key] = batch_list.index(value)

            if should_process:
                if should_crop_to_rectangle:
                    image = tileClient.getRegion(
                        datasetId, frame=i, left=left, top=top, right=right, bottom=bottom, units="base_pixels").squeeze()
                    print("image.shape", image.shape)
                else:
                    image = tileClient.getRegion(datasetId, frame=i).squeeze()
                sink.addTile(image, 0, 0, **new_params)

            sendProgress(i / len(tileClient.tiles['frames']), 'Crop',
                         f"Processing frame {i+1}/{len(tileClient.tiles['frames'])}")
    else:
        if should_crop_to_rectangle:
            image = tileClient.getRegion(
                datasetId, frame=0, left=left, top=top, right=right, bottom=bottom, units="base_pixels").squeeze()
        else:
            image = tileClient.getRegion(datasetId, frame=0).squeeze()

        sink.addTile(image, 0, 0, z=0)  # X, Y, Z

    # Copy over the metadata
    if 'channels' in tileClient.tiles:
        sink.channelNames = [tileClient.tiles['channels'][i] for i in channels]

    sink.mm_x = tileClient.tiles['mm_x']
    sink.mm_y = tileClient.tiles['mm_y']
    sink.magnification = tileClient.tiles['magnification']
    print("sink.getMetadata()", sink.getMetadata())
    sink.write('/tmp/cropped.tiff')
    print("Wrote to file")

    item = gc.uploadFileToFolder(datasetId, '/tmp/cropped.tiff')
    metadata = {
        'tool': 'Crop',
    }
    if should_crop_to_rectangle:
        metadata['crop left'] = left
        metadata['crop top'] = top
        metadata['crop right'] = right
        metadata['crop bottom'] = bottom
    gc.addMetadataToItem(item['itemId'], metadata)
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
