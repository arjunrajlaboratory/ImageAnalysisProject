import argparse
import json
import sys
import os
from functools import partial
from itertools import product
import uuid
import pprint
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
        'Connect sequentially': {
            'type': 'checkbox',
            'default': True,
            'displayOrder': 8
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'displayOrder': 9,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.3,
            'displayOrder': 10,
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


def assign_temporary_ids(annotations):
    """
    Assigns a unique temporary ID to each annotation.

    Args:
        annotations (list): List of annotation dictionaries.

    Returns:
        list: List of annotations with added 'tempId'.
    """
    for ann in annotations:
        ann['tempId'] = str(uuid.uuid4())
    return annotations


def assign_parent_ids(annotations, parent_annotations):
    """
    Assigns the 'parentId' to each annotation based on the corresponding parent annotation.

    Args:
        annotations (list): List of child annotation dictionaries.
        parent_annotations (list): List of parent annotation dictionaries.

    Returns:
        list: List of child annotations with added 'parentId'.
    """
    if len(annotations) != len(parent_annotations):
        raise ValueError("The number of annotations and parent_annotations must be the same.")

    for child_ann, parent_ann in zip(annotations, parent_annotations):
        # Attempt to get 'tempId'; if not present, fallback to '_id'; else None
        child_ann['parentId'] = parent_ann.get('tempId') or parent_ann.get('_id', None)
    return annotations


def strip_ids(annotations):
    """
    Strips 'tempId' and 'parentId' from each annotation.

    Args:
        annotations (list): List of annotation dictionaries with 'tempId' and 'parentId'.

    Returns:
        list: List of annotations without 'tempId' and 'parentId'.
        list: List of dictionaries containing 'tempId' and 'parentId' for mapping.
    """
    # stripped_annotations = []
    id_mappings = []

    for ann in annotations:
        temp_id = ann.pop('tempId', None)
        parent_id = ann.pop('parentId', None)
        # stripped_annotations.append(ann)
        id_mappings.append({
            'tempId': temp_id,
            'parentId': parent_id
        })

    return annotations, id_mappings
    # return stripped_annotations, id_mappings


def generate_connections(annotations_from_server, id_mappings, datasetId, tags, propagation_direction):
    """
    Generates a list of connection dictionaries mapping parent IDs to child IDs.

    Args:
        annotations_from_server (list): List of annotation dictionaries returned from the server with '_id'.
        id_mappings (list): List of dictionaries containing 'tempId' and 'parentId'.

    Returns:
        list: List of connection dictionaries with 'parentId' and 'childId'.
    """
    if len(annotations_from_server) != len(id_mappings):
        raise ValueError("The number of annotations_from_server and id_mappings must be the same.")

    # Create a mapping from tempId to server-assigned _id
    temp_to_server_id = {}
    for server_ann, mapping in zip(annotations_from_server, id_mappings):
        temp_id = mapping.get('tempId')
        server_id = server_ann.get('_id')
        if temp_id and server_id:
            temp_to_server_id[temp_id] = server_id

    # Generate connections
    connections = []
    for mapping in id_mappings:
        parent_temp_id = mapping.get('parentId')
        child_temp_id = mapping.get('tempId')

        parent_server_id = temp_to_server_id.get(parent_temp_id, parent_temp_id) # If the parent_temp_id is not in the temp_to_server_id, then use the parent_temp_id as the server_id, which should be the original annotationId.
        if parent_server_id is None: # If there is truly no match, either mapped or original, then skip.
            continue
        child_server_id = temp_to_server_id.get(child_temp_id)

        if child_server_id:
            if propagation_direction == 'Forward': # Always make the earlier in Time/Z be the parent
                connections.append({
                    'parentId': parent_server_id,
                    'childId': child_server_id,
                    'tags': tags,
                    'datasetId': datasetId
                })
            else:
                connections.append({
                    'parentId': child_server_id,
                    'childId': parent_server_id,
                    'tags': tags,
                    'datasetId': datasetId
                })

    return connections


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
    connect_sequentially = params['workerInterface']['Connect sequentially']
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
            # Slightly inefficient because we are loading the image twice, but it's not a big deal.

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
            temp_annotations = assign_temporary_ids(temp_annotations) # Assign a temporary id to each annotation
            # In this case, we do not need to assign parentIds, because these are the "root" annotations
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
        temp_annotations = assign_temporary_ids(temp_annotations) # Assign a temporary id to each annotation
        if not resegment_propagation_objects:
            sliced_new_annotations = assign_temporary_ids(sliced_new_annotations)
            sliced_annotations.extend(sliced_new_annotations)
            temp_annotations = assign_parent_ids(temp_annotations, sliced_annotations) # Assign a parentId to each annotation
        else:
            temp_annotations = assign_parent_ids(temp_annotations, sliced_new_annotations) # Assign a parentId to each annotation
        new_annotations.extend(temp_annotations)

        # Update progress after each batch
        processed_batches += 1
        fraction_done = processed_batches / total_batches
        sendProgress(fraction_done, "Propagating annotations", f"{processed_batches} of {total_batches} frames processed")

    sendProgress(0.9, "Uploading annotations", f"Sending {len(new_annotations)} annotations to server")

    stripped_annotations, id_mappings = strip_ids(new_annotations)
    pprint.pprint(stripped_annotations)
    pprint.pprint(id_mappings)

    annotations_from_server = annotationClient.createMultipleAnnotations(stripped_annotations)

    if connect_sequentially:
        connections = generate_connections(annotations_from_server, id_mappings, datasetId, ['SAM2_PROPAGATED'], propagation_direction)
        pprint.pprint(connections)
        annotationClient.createMultipleConnections(connections)
    


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
