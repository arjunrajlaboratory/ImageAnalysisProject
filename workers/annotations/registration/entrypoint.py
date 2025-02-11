import base64
import argparse
import json
import sys
import pprint

from operator import itemgetter

import annotation_client.tiles as tiles
import annotation_client.workers as workers
import annotation_client.annotations as annotations

from annotation_client.utils import sendProgress, sendError, sendWarning

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
                'placeholder': 'ex. 8; default is 1',
                'label': 'Reference Z coordinate.',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Z coordinate to reference for registration. Default is 1.'
            },
            'displayOrder': 2
        },
        'Reference Time Coordinate': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 8; default is 1',
                'label': 'Reference Time coordinate.',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the time point to reference for registration. Default is 1.'
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
        'Control point tag': {
            'type': 'tags',
            'tooltip': 'Enter the tag of the control points to use for registration.',
            'displayOrder': 7
        },
        'Apply algorithm after control points': {
            'type': 'checkbox',
            'tooltip': 'Apply the registration algorithm after the control points are applied.',
            'default': False,
            'displayOrder': 8
        },
        'Algorithm': {
            'type': 'select',
            'items': ['None (control points only)', 'Translation', 'Rigid', 'Affine'],
            'label': 'Algorithm',
            'default': 'Translation',
            'tooltip': 'Select the registration constraints (pystackreg). If you just want to use control points, select "None (control points only)".\n'
                       'Translation: Only translation is applied.\n'
                       'Rigid: Translation and rotation are applied.\n'
                       'Affine: Translation, rotation, and scaling are applied.',
            'displayOrder': 9
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def safe_astype(arr, dtype):
    """
    This function is used to cast an array to a new dtype.
    It will clip the values to the range of the new dtype if necessary.
    This is useful for avoiding overflow when casting to integer types.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(arr, info.min, info.max).astype(dtype)
    return arr.astype(dtype)


def register_images(image1, image2, algorithm, sr):
    if algorithm == 'None (control points only)':
        return np.eye(3)
    else:
        return sr.register(image1, image2)


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
    apply_algorithm_after_control_points = params['workerInterface']['Apply algorithm after control points']

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

    if workerInterface['Reference Z Coordinate'] == "":
        reference_Z = 0
    else:
        reference_Z = int(workerInterface['Reference Z Coordinate']) - 1
        if reference_Z > 0:
            # Check if IndexZ exists and if so, check that the reference Z is within the range
            if 'IndexZ' in tileInfo['IndexRange']:
                if reference_Z >= tileInfo['IndexRange']['IndexZ']:
                    sendError(f"Reference Z {reference_Z+1} "
                              "is out of range.")
                    return
            else:
                sendError("IndexZ not found in tileInfo")
                return
    if workerInterface['Reference Time Coordinate'] == "":
        reference_Time = 0
    else:
        reference_Time = int(workerInterface['Reference Time Coordinate']) - 1
        if reference_Time > 0:
            # Check if IndexT exists and if so, check that the reference time is within the range
            if 'IndexT' in tileInfo['IndexRange']:
                if reference_Time >= tileInfo['IndexRange']['IndexT']:
                    sendError(f"Reference time {reference_Time+1} "
                              "is out of range.")
                    return
            else:
                sendError("IndexT not found in tileInfo")
                return

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
            blobAnnotationList, params['workerInterface']['Reference region tag'], exclusive=False)
        if cropAnnotationList is None or len(cropAnnotationList) == 0:
            sendError("No reference region found")
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

    should_use_control_points = (workerInterface['Control point tag'] is not None and
                                 workerInterface['Control point tag'])
    cp_dict = {}
    if should_use_control_points:
        control_point_annotations = annotationClient.getAnnotationsByDatasetId(
            datasetId, limit=1000, shape='point')
        control_point_annotations = annotation_tools.get_annotations_with_tags(
            control_point_annotations, workerInterface['Control point tag'], exclusive=False)
        if len(control_point_annotations) == 0:
            sendWarning("No control points found")
        else:
            # Build a dictionary mapping (XY, Time) to (x, y) using the first control point found
            for cp in control_point_annotations:
                loc = cp.get('location', {})
                cp_xy = loc.get('XY')
                cp_time = loc.get('Time')
                if cp_xy is not None and cp_time is not None:
                    if (cp_xy, cp_time) not in cp_dict:
                        coords = cp.get('coordinates')
                        if coords and len(coords) > 0:
                            x = coords[0].get('x')
                            y = coords[0].get('y')
                            if x is not None and y is not None:
                                cp_dict[(cp_xy, cp_time)] = (x, y)
            # For debugging:
            print("Control point dictionary:")
            pprint.pprint(cp_dict)

    # Initialize the stackreg object
    # TODO: Make this configurable based on selected algorithm
    if algorithm == 'None (control points only)':
        sr = StackReg(StackReg.TRANSLATION)
    elif algorithm == 'Translation':
        sr = StackReg(StackReg.TRANSLATION)
    elif algorithm == 'Rigid':
        sr = StackReg(StackReg.RIGID_BODY)
    elif algorithm == 'Affine':
        sr = StackReg(StackReg.AFFINE)
    else:
        sendError(f"Invalid algorithm: {algorithm}")
        return

    # Now let's compute the registration matrices
    registration_matrices = {}
    progress_counter = 0
    total_progress = len(apply_XY) * tileInfo['IndexRange']['IndexT']

    for xy in apply_XY:
        # Start with the identity matrix at t=0.
        registration_matrices[(xy, 0)] = np.eye(3)
        frame = tileClient.coordinatesToFrameIndex(
            xy, reference_Z, 0, reference_channel)
        if should_use_reference_region:
            current_image = tileClient.getRegion(
                datasetId, frame=frame,
                left=reference_region_left,
                top=reference_region_top,
                right=reference_region_right,
                bottom=reference_region_bottom,
                units="base_pixels").squeeze()
        else:
            current_image = tileClient.getRegion(
                datasetId, frame=frame).squeeze()

        for t in range(1, tileInfo['IndexRange']['IndexT']):
            next_frame = tileClient.coordinatesToFrameIndex(
                xy, reference_Z, t, reference_channel)
            if should_use_reference_region:
                next_image = tileClient.getRegion(
                    datasetId, frame=next_frame,
                    left=reference_region_left,
                    top=reference_region_top,
                    right=reference_region_right,
                    bottom=reference_region_bottom,
                    units="base_pixels").squeeze()
            else:
                next_image = tileClient.getRegion(
                    datasetId, frame=next_frame).squeeze()

            # Check for control points at (xy, t-1) and (xy, t)
            cp_prev = cp_dict.get(
                (xy, t - 1)) if should_use_control_points else None
            cp_curr = cp_dict.get(
                (xy, t)) if should_use_control_points else None

            if cp_prev is not None and cp_curr is not None:
                # Compute the translation (control point) matrix.
                dx = cp_curr[0] - cp_prev[0]
                dy = cp_curr[1] - cp_prev[1]
                CP = np.array([[1, 0, dx],
                               [0, 1, dy],
                               [0, 0, 1]])
                if apply_algorithm_after_control_points:
                    # First apply the control point correction then do the registration algorithm.
                    reg_matrix = register_images(
                        current_image, sr.transform(next_image, tmat=CP), algorithm, sr)
                    combined_matrix = np.dot(reg_matrix, CP)
                else:
                    # Use only the control point transformation.
                    combined_matrix = CP
                cumulative_registration_matrix = np.dot(
                    combined_matrix, registration_matrices[(xy, t - 1)])
            else:
                # Fall back to using the algorithm's registration if control points are not available.
                reg_matrix = register_images(
                    current_image, next_image, algorithm, sr)
                cumulative_registration_matrix = np.dot(
                    reg_matrix, registration_matrices[(xy, t - 1)])

            registration_matrices[(xy, t)] = cumulative_registration_matrix
            current_image = next_image

            sendProgress(progress_counter / total_progress,
                         'Calculating registration matrices',
                         f"Processing t: {t}, xy: {xy}")
            progress_counter += 1

    # If a reference time other than t=0 was specified, adjust the matrices so that
    # the registration matrices are relative to the chosen reference time.
    if reference_Time != 0:
        for xy in apply_XY:
            # Check that the reference time is within the available range
            if reference_Time >= tileInfo['IndexRange']['IndexT']:
                sendError(f"Reference time {reference_Time+1} "
                          f"is out of range for XY coordinate {xy}.")
                return
            # Get the cumulative matrix at the reference time for this XY position.
            ref_matrix = registration_matrices[(xy, reference_Time)]
            try:
                inv_ref = np.linalg.inv(ref_matrix)
            except np.linalg.LinAlgError as e:
                sendError("Could not invert reference matrix for XY "
                          f"{xy} at time {reference_Time+1}: {str(e)}")
                return
            # Multiply all matrices by the inverse of the reference matrix.
            for t in range(tileInfo['IndexRange']['IndexT']):
                registration_matrices[(xy, t)] = np.dot(
                    inv_ref, registration_matrices[(xy, t)])

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
                        image = safe_astype(transformed_image, image.dtype)
                else:
                    transformed_image = sr.transform(
                        image, tmat=registration_matrices[(0, frame['IndexT'])])
                    image = safe_astype(transformed_image, image.dtype)

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
