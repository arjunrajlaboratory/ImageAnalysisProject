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
from sam2.build_sam import build_sam2_video_predictor

from PIL import Image

from annotation_client.utils import sendProgress

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # List all .pt files in the /sam2/checkpoints directory
    models = [f for f in os.listdir('/code/segment-anything-2-nimbus/checkpoints') if f.endswith('.pt')]

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
        'Track across': {
            'type': 'select',
            'items': ['Time', 'Z'],
            'default': 'Time',
            'displayOrder': 3
        },
        'Track direction': {
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
        'Tag of objects to track': {
            'type': 'tags',
            'displayOrder': 6
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'displayOrder': 7,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.3,
            'displayOrder': 8,
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

    annotationClient = annotations_client.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    model = params['workerInterface']['Model']
    padding = float(params['workerInterface']['Padding'])
    smoothing = float(params['workerInterface']['Smoothing'])
    track_tags = params['workerInterface']['Tag of objects to track']
    track_across = params['workerInterface']['Track across']
    track_direction = params['workerInterface']['Track direction']
    batch_xy = params['workerInterface']['Batch XY']
    batch_z = params['workerInterface']['Batch Z']
    batch_time = params['workerInterface']['Batch Time']

    # Here's some code to set up the model and predictor.
    # use bfloat16 for computation
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_path = f"/code/segment-anything-2-nimbus/checkpoints/{model}"
    # This needless naming change is making me sad.
    model_to_cfg = {
        'sam2.1_hiera_base_plus.pt': 'sam2.1_hiera_b+.yaml',
        'sam2.1_hiera_large.pt': 'sam2.1_hiera_l.yaml',
        'sam2.1_hiera_small.pt': 'sam2.1_hiera_s.yaml',
        'sam2.1_hiera_tiny.pt': 'sam2.1_hiera_t.yaml',
    }
    model_cfg = f"configs/sam2.1/{model_to_cfg[model]}"
    # sam2_model = build_sam2(model_cfg, checkpoint_path, device='cuda', apply_postprocessing=False)  # device='cuda' for GPU
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device="cuda")  # device="cuda" for GPU

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
    # If the propagation_direction is backward, then we need to reverse the variable specified by track_across.
    if track_direction == 'Backward':
        if track_across == 'Time':
            batch_time = batch_time[::-1]
        elif track_across == 'Z':
            batch_z = batch_z[::-1]

    # Get the annotations with the track_tags, because those are the ones we want to propagate.
    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    annotationList = annotation_tools.get_annotations_with_tags(annotationList, track_tags, exclusive=False)

    new_annotations = []

    # Define total_batches for progress reporting
    if track_across == 'Time':
        total_batches = len(batch_xy) * len(batch_z)
    elif track_across == 'Z':
        total_batches = len(batch_xy) * len(batch_time)
    else:
        sendProgress(1, "Error", f"Invalid track across value: {track_across}")
        return

    processed_batches = 0
    inference_state = None

    if track_across == 'Time':
        # Build mappings between frame_idx and Time
        frame_idx_to_Time = {idx: Time for idx, Time in enumerate(batch_time)}
        Time_to_frame_idx = {Time: idx for idx, Time in enumerate(batch_time)}

        for XY in batch_xy:
            for Z in batch_z:
                
                sliced_annotations = [ann for ann in annotationList if ann['location']['XY'] == XY and ann['location']['Z'] == Z and ann['location']['Time'] in batch_time]
                if len(sliced_annotations) == 0:
                    # No annotations to propagate
                    processed_batches += 1
                    fraction_done = processed_batches / total_batches
                    sendProgress(fraction_done, "Tracking objects", f"{processed_batches} of {total_batches} batches processed")
                    continue

                # Prepare video_data
                batches = [(XY, Z, Time) for Time in batch_time]
                video_data = {
                    "tileClient": tileClient,
                    "datasetId": datasetId,
                    "batches": batches
                }

                # Reset and init_state
                if inference_state is not None:
                    predictor.reset_state(inference_state)
                inference_state = predictor.init_state(video_path=video_data)

                # For each annotation, add prompt at its Time (ann_frame_idx)
                for ann_obj_id, ann in enumerate(sliced_annotations):
                    Time_ann = ann['location']['Time']
                    if Time_ann not in Time_to_frame_idx:
                        continue  # Skip annotations not in batch_time
                    ann_frame_idx = Time_to_frame_idx[Time_ann]
                    # Convert annotation to polygon and box
                    polygon = annotation_tools.annotations_to_polygons(ann)[0]
                    box = np.array(polygon.bounds, dtype=np.float32)
                    # Add box prompt
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=box,
                    )

                # Run predictor
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    # Collect masks
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # Convert masks to annotations
                for frame_idx, masks in video_segments.items():
                    Time_frame = frame_idx_to_Time[frame_idx]
                    for obj_id, mask in masks.items():
                        # Convert mask to polygon
                        contours = find_contours(mask.squeeze(), 0.5)
                        if len(contours) == 0:
                            continue
                        polygon = Polygon(contours[0]).buffer(padding).simplify(smoothing, preserve_topology=True)
                        # Create annotation
                        annotation = annotation_tools.polygons_to_annotations(polygon, datasetId, XY=XY, Z=Z, Time=Time_frame, tags=tags, channel=channel)
                        new_annotations.extend(annotation)

                # Update progress
                processed_batches += 1
                fraction_done = processed_batches / total_batches
                sendProgress(fraction_done, "Tracking objects", f"{processed_batches} of {total_batches} batches processed")

    elif track_across == 'Z':
        # Build mappings between frame_idx and Z
        frame_idx_to_Z = {idx: Z for idx, Z in enumerate(batch_z)}
        Z_to_frame_idx = {Z: idx for idx, Z in enumerate(batch_z)}

        for XY in batch_xy:
            for Time in batch_time:
                # Get annotations in batch_z for this XY and Time
                sliced_annotations = [ann for ann in annotationList if ann['location']['XY'] == XY and ann['location']['Time'] == Time and ann['location']['Z'] in batch_z]
                if len(sliced_annotations) == 0:
                    # No annotations to propagate
                    processed_batches += 1
                    fraction_done = processed_batches / total_batches
                    sendProgress(fraction_done, "Tracking objects", f"{processed_batches} of {total_batches} batches processed")
                    continue

                # Prepare video_data
                batches = [(XY, Z, Time) for Z in batch_z]
                video_data = {
                    "tileClient": tileClient,
                    "datasetId": datasetId,
                    "batches": batches
                }

                # Reset and init_state
                if inference_state is not None:
                    predictor.reset_state(inference_state)
                inference_state = predictor.init_state(video_path=video_data)

                # For each annotation, add prompt at its Z (ann_frame_idx)
                for ann_obj_id, ann in enumerate(sliced_annotations):
                    Z_ann = ann['location']['Z']
                    if Z_ann not in Z_to_frame_idx:
                        continue  # Skip annotations not in batch_z
                    ann_frame_idx = Z_to_frame_idx[Z_ann]
                    # Convert annotation to polygon and box
                    polygon = annotation_tools.annotations_to_polygons(ann)[0]
                    box = np.array(polygon.bounds, dtype=np.float32)
                    # Add box prompt
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        box=box,
                    )

                # Run predictor
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    # Collect masks
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # Convert masks to annotations
                for frame_idx, masks in video_segments.items():
                    Z_frame = frame_idx_to_Z[frame_idx]
                    for obj_id, mask in masks.items():
                        # Convert mask to polygon
                        contours = find_contours(mask.squeeze(), 0.5)
                        if len(contours) == 0:
                            continue
                        polygon = Polygon(contours[0]).buffer(padding).simplify(smoothing, preserve_topology=True)
                        # Create annotation
                        annotation = annotation_tools.polygons_to_annotations(polygon, datasetId, XY=XY, Z=Z_frame, Time=Time, tags=tags, channel=channel)
                        new_annotations.extend(annotation)

                # Update progress
                processed_batches += 1
                fraction_done = processed_batches / total_batches
                sendProgress(fraction_done, "Tracking objects", f"{processed_batches} of {total_batches} batches processed")

    else:
        sendProgress(1, "Error", f"Invalid track across value: {track_across}")
        return

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
