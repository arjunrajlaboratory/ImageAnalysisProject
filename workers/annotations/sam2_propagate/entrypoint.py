import argparse
import json
import sys
import os
from functools import partial
from itertools import product

import annotation_client.annotations as annotations_client
import annotation_client.workers as workers
import annotation_client.tiles as tiles

import annotation_utilities.annotation_tools as annotation_tools
import annotation_utilities.batch_argument_parser as batch_argument_parser

import numpy as np  # library for array manipulation
from shapely.geometry import Polygon
from skimage.measure import find_contours
from shapely.geometry import Polygon

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

from annotation_client.utils import sendProgress

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # List all .pt files in the /sam2/checkpoints directory
    models = [f for f in os.listdir('/code/sam2/checkpoints') if f.endswith('.pt')]

    # Set the default model
    default_model = 'sam2.1_hiera_small.pt' if 'sam2.1_hiera_small.pt' in models else models[0] if models else None

    # Available types: number, text, tags, layer
    interface = {
        'Batch XY': {
            'type': 'text',
            'displayOrder': 0
        },
        'Batch Z': {
            'type': 'text',
            'displayOrder': 1
        },
        'Batch Time': {
            'type': 'text',
            'displayOrder': 2
        },
        'Propagate across': {
            'type': 'select',
            'items': ['Time', 'Z'],
            'default': 'Time',
            'displayOrder': 3
        },
        'Propagation direction': {
            'type': 'select',
            'items': ['Forward', 'Backward'],
            'default': 'Forward',
            'displayOrder': 4
        },
        'Model': {
            'type': 'select',
            'items': models,
            'default': default_model,
            'displayOrder': 5
        },
        'Tag of objects to propagate': {
            'type': 'tags',
            'displayOrder': 6
        },
        'Resegment propagation objects': {
            'type': 'checkbox',
            'default': True,
            'displayOrder': 7
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'displayOrder': 8,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.3,
            'displayOrder': 9,
        },
    }
    # Send the interface object to the server
    client.setWorkerImageInterface(image, interface)


def sam2_masks_to_polygons(masks, smoothing, padding):
    """
    Convert SAM2 masks to simplified polygons.

    Args:
    masks (numpy.ndarray): Array of masks from SAM2 prediction.
    smoothing (float): Smoothing factor for polygon simplification.

    Returns:
    list: List of simplified Shapely Polygon objects.
    """
    polygons = []
    for mask in masks:
        # Sometimes you get multiple masks (from multiple boxes), in which case you'll need to squeeze it.
        # But if you only get one mask, don't squeeze it.
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        contours = find_contours(mask, 0.5)
        if contours:  # Check if contours were found
            polygon = Polygon(contours[0]).simplify(smoothing, preserve_topology=True)
            polygon = polygon.buffer(padding)
            polygons.append(polygon)
    return polygons


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

    annotationClient = annotations_client.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    model = params['workerInterface']['Model']
    padding = float(params['workerInterface']['Padding'])
    smoothing = float(params['workerInterface']['Smoothing'])
    propagate_tags = params['workerInterface']['Tag of objects to propagate']
    resegment_propagation_objects = params['workerInterface']['Resegment propagation objects']
    propagate_across = params['workerInterface']['Propagate across']
    propagation_direction = params['workerInterface']['Propagation direction']
    batch_xy = params['workerInterface']['Batch XY']
    batch_z = params['workerInterface']['Batch Z']
    batch_time = params['workerInterface']['Batch Time']

    batch_xy = batch_argument_parser.process_range_list(batch_xy, convert_one_to_zero_index=True)
    batch_z = batch_argument_parser.process_range_list(batch_z, convert_one_to_zero_index=True)
    batch_time = batch_argument_parser.process_range_list(batch_time, convert_one_to_zero_index=True)

    tile = params['tile']
    channel = params['channel']
    tags = params['tags']

    XY = tile['XY']
    Z = tile['Z']
    Time = tile['Time']

    if batch_xy is None:
        batch_xy = [tile['XY']]
    if batch_z is None:
        batch_z = [tile['Z']]
    if batch_time is None:
        batch_time = [tile['Time']]

    batch_xy = list(batch_xy)
    batch_z = list(batch_z)
    batch_time = list(batch_time)

    # If the propagation_direction is forward, then we are fine.
    # If the propagation_direction is backward, then we need to reverse the variable specified by propagate_across.
    if propagation_direction == 'Backward':
        if propagate_across == 'Time':
            batch_time = list(reversed(list(batch_time)))
        elif propagate_across == 'Z':
            batch_z = list(reversed(list(batch_z)))

    batches = list(product(batch_xy, batch_z, batch_time))
    total_batches = len(batches)
    processed_batches = 0

    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, propagate_tags, exclusive=False)

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_path = f"/code/sam2/checkpoints/{model}"
    # This needless naming change is making me sad.
    model_to_cfg = {
        'sam2.1_hiera_base_plus.pt': 'sam2.1_hiera_b+.yaml',
        'sam2.1_hiera_large.pt': 'sam2.1_hiera_l.yaml',
        'sam2.1_hiera_small.pt': 'sam2.1_hiera_s.yaml',
        'sam2.1_hiera_tiny.pt': 'sam2.1_hiera_t.yaml',
    }
    model_cfg = f"configs/sam2.1/{model_to_cfg[model]}"
    sam2_model = build_sam2(model_cfg, checkpoint_path, device='cuda', apply_postprocessing=False)  # device='cuda' for GPU
    predictor = SAM2ImagePredictor(sam2_model)

    rangeXY = tileClient.tiles['IndexRange'].get('IndexXY', 1)
    rangeZ = tileClient.tiles['IndexRange'].get('IndexZ', 1)
    rangeTime = tileClient.tiles['IndexRange'].get('IndexT', 1)

    new_annotations = []

    for batch in batches:
        XY, Z, Time = batch
        # Search the annotationList for annotations with the current Time, XY, and Z.
        sliced_annotations = annotation_tools.filter_elements_T_XY_Z(annotationList, Time, XY, Z)
        if resegment_propagation_objects and len(sliced_annotations) > 0:
            # If we are resegmenting, then we will load the current image and run the predictor.
            # Slight inefficient because we are loading the image twice, but it's not a big deal.

            # Get the images for the current Time and XY and Z
            images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, XY, Z, Time)
            layers = annotation_tools.get_layers(tileClient.client, datasetId)

            merged_image = annotation_tools.process_and_merge_channels(images, layers)
            image = merged_image.astype(np.float32)

            predictor.set_image(image)

            polygons = annotation_tools.annotations_to_polygons(sliced_annotations)
            boxes = [polygon.bounds for polygon in polygons]
            input_boxes = np.array(boxes)

            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            # Find contours in the mask
            temp_polygons = sam2_masks_to_polygons(masks, smoothing, padding)
            temp_annotations = annotation_tools.polygons_to_annotations(temp_polygons, datasetId, XY=XY, Time=Time, Z=Z, tags=tags, channel=channel)
            new_annotations.extend(temp_annotations)  # Add the new annotations to the new_annotations list for downstream processing

        sliced_new_annotations = annotation_tools.filter_elements_T_XY_Z(new_annotations, Time, XY, Z)  # This will include the resegmented annotations if they were created

        if len(sliced_annotations) == 0 and len(sliced_new_annotations) == 0: # If we didn't find any annotations to propagate, skip
            continue

        # Propose a location to look for the next image, depending on the propagation_direction and the propagate_across variable (either Time or Z).
        if propagate_across == 'Time':
            # find the current index of Time in the batch_time list
            current_index = batch_time.index(Time)
            if not current_index == len(batch_time) - 1: # If we're not at the last frame, propagate to the next frame
                next_Time = batch_time[current_index + 1]
                next_Z = Z
            else:
                continue
        elif propagate_across == 'Z':
            current_index = batch_z.index(Z)
            if not current_index == len(batch_z) - 1:
                next_Z = batch_z[current_index + 1]
                next_Time = Time
            else:
                continue

        # Get the images for the next Time and XY and Z
        images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, XY, next_Z, next_Time)
        layers = annotation_tools.get_layers(tileClient.client, datasetId)

        merged_image = annotation_tools.process_and_merge_channels(images, layers)
        image = merged_image.astype(np.float32)

        predictor.set_image(image)

        polygons = []
        if not resegment_propagation_objects:
            # If we already resegmented the annotations, then the new ones are already in sliced_new_annotations
            polygons = annotation_tools.annotations_to_polygons(sliced_annotations)
        polygons.extend(annotation_tools.annotations_to_polygons(sliced_new_annotations))
        boxes = [polygon.bounds for polygon in polygons]
        input_boxes = np.array(boxes)

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Find contours in the mask
        temp_polygons = sam2_masks_to_polygons(masks, smoothing, padding)

        temp_annotations = annotation_tools.polygons_to_annotations(temp_polygons, datasetId, XY=XY, Time=next_Time, Z=next_Z, tags=tags, channel=channel)
        new_annotations.extend(temp_annotations)

        # Update progress after each batch
        processed_batches += 1
        fraction_done = processed_batches / total_batches
        sendProgress(fraction_done, "Propagating annotations", f"{processed_batches} of {total_batches} frames processed")

    sendProgress(0.9, "Uploading annotations", f"Sending {len(new_annotations)} annotations to server")
    annotationClient.createMultipleAnnotations(new_annotations)


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='SAM Automatic Mask Generator')

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
