import argparse
import json
import sys
import os
from collections import defaultdict
from itertools import product

import annotation_client.annotations as annotations_client
import annotation_client.workers as workers
import annotation_client.tiles as tiles

import annotation_utilities.annotation_tools as annotation_tools
import annotation_utilities.batch_argument_parser as batch_argument_parser

import numpy as np
from shapely.geometry import Polygon
from skimage.measure import find_contours

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from annotation_client.utils import sendProgress


def interface(image, apiUrl, token):
    """
    Define the user interface for the SAM2 Refiner tool.
    """
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # List all .pt files in the /sam2/checkpoints directory
    models = [f for f in os.listdir('/code/sam2/checkpoints') if f.endswith('.pt')]

    # Set the default model
    default_model = 'sam2.1_hiera_small.pt' if 'sam2.1_hiera_small.pt' in models else models[0] if models else None

    interface = {
        'SAM2 Refiner': {
            'type': 'notes',
            'value': 'This tool uses the SAM2 model to refine existing annotations. '
                     'It takes existing annotations, uses them as prompts (via bounding boxes), '
                     'and generates refined segmentation masks that better match the underlying image data.',
            'displayOrder': 0,
        },
        'Batch XY': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the XY positions you want to process',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the XY positions to process. Separate multiple groups with a comma.'
            },
            'displayOrder': 1
        },
        'Batch Z': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the Z positions you want to process',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Z positions to process. Separate multiple groups with a comma.'
            },
            'displayOrder': 2
        },
        'Batch Time': {
            'type': 'text',
            'vueAttrs': {
                'placeholder': 'ex. 1-3, 5-8',
                'label': 'Enter the Time positions you want to process',
                'persistentPlaceholder': True,
                'filled': True,
                'tooltip': 'Enter the Time positions to process. Separate multiple groups with a comma.'
            },
            'displayOrder': 3
        },
        'Tag of objects to refine': {
            'type': 'tags',
            'displayOrder': 4
        },
        'Model': {
            'type': 'select',
            'items': models,
            'default': default_model,
            'tooltip': 'The SAM2 model to use for refinement. Larger models may give slightly better results, '
                       'but are slower and use more memory.',
            'displayOrder': 5
        },
        'Delete original annotations': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'If checked, the original annotations will be deleted after refinement.',
            'displayOrder': 6
        },
        'Padding': {
            'type': 'number',
            'min': -20,
            'max': 20,
            'default': 0,
            'unit': 'Pixels',
            'tooltip': 'Padding will expand (or, if negative, contract) the polygon boundary.',
            'displayOrder': 7,
        },
        'Smoothing': {
            'type': 'number',
            'min': 0,
            'max': 3,
            'default': 0.7,
            'tooltip': 'Smoothing is used to simplify the polygons; a value of 0.7 is a good default.',
            'displayOrder': 8,
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
        padding (float): Padding to expand/contract polygons.

    Returns:
        list: List of simplified Shapely Polygon objects.
    """
    polygons = []
    for mask in masks:
        # Handle different mask dimensions
        if mask.ndim == 3:
            mask = mask.squeeze(0)

        contours = find_contours(mask, 0.5)
        if contours:  # Check if contours were found
            try:
                polygon = Polygon(contours[0]).simplify(smoothing, preserve_topology=True)
                if padding != 0:
                    polygon = polygon.buffer(padding)
                if not polygon.is_empty and polygon.is_valid:
                    polygons.append(polygon)
            except Exception as e:
                print(f"Warning: Failed to create polygon from contour: {e}")
                continue

    return polygons


def group_annotations_by_location(annotations):
    """
    Group annotations by their location (XY, Z, Time) to minimize image loading.
    
    Args:
        annotations (list): List of annotation dictionaries.
    
    Returns:
        dict: Dictionary with (XY, Z, Time) tuples as keys and lists of annotations as values.
    """
    grouped = defaultdict(list)
    for ann in annotations:
        location = ann.get('location', {})
        key = (
            location.get('XY', 0),
            location.get('Z', 0),
            location.get('Time', 0)
        )
        grouped[key].append(ann)
    return grouped


def compute(datasetId, apiUrl, token, params):
    """
    Main compute function for SAM2 Refiner.
    
    Refines existing annotations by using them as prompts for SAM2 segmentation.
    """

    # Initialize clients
    annotationClient = annotations_client.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    # Get parameters from interface
    model = params['workerInterface']['Model']
    padding = float(params['workerInterface']['Padding'])
    smoothing = float(params['workerInterface']['Smoothing'])
    refine_tags = params['workerInterface']['Tag of objects to refine']
    delete_original = params['workerInterface']['Delete original annotations']
    batch_xy = params['workerInterface'].get('Batch XY', '')
    batch_z = params['workerInterface'].get('Batch Z', '')
    batch_time = params['workerInterface'].get('Batch Time', '')

    # Parse batch parameters
    batch_xy = batch_argument_parser.process_range_list(batch_xy, convert_one_to_zero_index=True)
    batch_z = batch_argument_parser.process_range_list(batch_z, convert_one_to_zero_index=True)
    batch_time = batch_argument_parser.process_range_list(batch_time, convert_one_to_zero_index=True)

    # Get annotation context
    tile = params['tile']
    channel = params['channel']
    tags = params['tags']

    # Use tile values as defaults if batch parameters not specified
    if batch_xy is None:
        batch_xy = [tile['XY']]
    if batch_z is None:
        batch_z = [tile['Z']]
    if batch_time is None:
        batch_time = [tile['Time']]

    batch_xy = list(batch_xy)
    batch_z = list(batch_z)
    batch_time = list(batch_time)

    sendProgress(0.05, "Fetching annotations", "Retrieving annotations to refine")

    # Get all polygon annotations with specified tags
    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    
    # Also try to get blob annotations if they exist
    try:
        blob_annotations = workerClient.get_annotation_list_by_shape('blob', limit=0)
        annotationList.extend(blob_annotations)
    except Exception:
        # Blob shape might not be supported on all servers
        pass
    
    # Filter by tags if specified
    if refine_tags:
        annotationList = annotation_tools.get_annotations_with_tags(annotationList, refine_tags, exclusive=False)

    # Filter annotations to only include those in our batch ranges
    filtered_annotations = []
    for ann in annotationList:
        location = ann.get('location', {})
        if (location.get('XY', 0) in batch_xy and 
            location.get('Z', 0) in batch_z and 
            location.get('Time', 0) in batch_time):
            filtered_annotations.append(ann)

    if len(filtered_annotations) == 0:
        sendProgress(1.0, "Complete", "No annotations found to refine")
        return

    # Group annotations by location to minimize image loading
    grouped_annotations = group_annotations_by_location(filtered_annotations)
    total_locations = len(grouped_annotations)

    sendProgress(0.1, "Initializing SAM2", f"Loading {model}")

    # Setup GPU acceleration if available
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    if use_cuda:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # Enable TF32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Setup SAM2 model
    checkpoint_path = f"/code/sam2/checkpoints/{model}"
    model_to_cfg = {
        'sam2.1_hiera_base_plus.pt': 'sam2.1_hiera_b+.yaml',
        'sam2.1_hiera_large.pt': 'sam2.1_hiera_l.yaml',
        'sam2.1_hiera_small.pt': 'sam2.1_hiera_s.yaml',
        'sam2.1_hiera_tiny.pt': 'sam2.1_hiera_t.yaml',
    }
    model_cfg = f"configs/sam2.1/{model_to_cfg[model]}"

    try:
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device, apply_postprocessing=False)
        predictor = SAM2ImagePredictor(sam2_model)
    except Exception as e:
        sendProgress(1.0, "Error", f"Failed to initialize SAM2 model: {str(e)}")
        return

    # Get layers for image processing
    layers = annotation_tools.get_layers(tileClient.client, datasetId)

    new_annotations = []
    annotations_to_delete = []
    processed_locations = 0
    total_annotations_processed = 0
    total_annotations = len(filtered_annotations)

    # Process each location group
    for (XY, Z, Time), location_annotations in grouped_annotations.items():
        processed_locations += 1
        sendProgress(
            0.2 + (0.6 * processed_locations / total_locations),
            "Processing locations",
            f"Location {processed_locations}/{total_locations} (XY:{XY+1}, Z:{Z+1}, Time:{Time+1})"
        )

        try:
            # Load image for this location once
            images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, XY, Z, Time)
            merged_image = annotation_tools.process_and_merge_channels(images, layers)
            image = merged_image.astype(np.float32)

            predictor.set_image(image)

            # Convert annotations to polygons and get bounding boxes
            polygons = annotation_tools.annotations_to_polygons(location_annotations)
            if not polygons:
                continue

            boxes = [polygon.bounds for polygon in polygons]

            # Process each box individually (correct SAM2 API usage)
            masks = []
            for i, box in enumerate(boxes):
                try:
                    # SAM2 expects box as [x_min, y_min, x_max, y_max]
                    input_box = np.array(box, dtype=np.float32)
                    
                    temp_masks, scores, logits = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box,  # Single box
                        multimask_output=False,
                    )
                    masks.append(temp_masks[0])

                    total_annotations_processed += 1

                except Exception as e:
                    print(f"Warning: Failed to process annotation {i+1}/{len(boxes)}: {e}")
                    continue

            if not masks:
                continue

            # Convert masks to polygons
            refined_polygons = sam2_masks_to_polygons(masks, smoothing, padding)

            if refined_polygons:
                # Create new annotations from refined polygons
                refined_annotations = annotation_tools.polygons_to_annotations(
                    refined_polygons, 
                    datasetId, 
                    XY=XY, 
                    Time=Time, 
                    Z=Z, 
                    tags=tags, 
                    channel=channel
                )

                new_annotations.extend(refined_annotations)

            # Track original annotations for deletion if requested
            if delete_original:
                annotations_to_delete.extend([ann['_id'] for ann in location_annotations if '_id' in ann])

        except Exception as e:
            print(f"Error processing location (XY:{XY}, Z:{Z}, Time:{Time}): {e}")
            continue

    # Upload new annotations
    if len(new_annotations) > 0:
        sendProgress(0.85, "Uploading annotations", f"Sending {len(new_annotations)} refined annotations to server")
        try:
            annotationClient.createMultipleAnnotations(new_annotations)
        except Exception as e:
            print(f"Error uploading annotations: {e}")

         # Delete original annotations if requested
    if delete_original and len(annotations_to_delete) > 0:
        sendProgress(0.95, "Cleaning up", f"Deleting {len(annotations_to_delete)} original annotations")
        try:
            annotationClient.deleteMultipleAnnotations(annotations_to_delete)
        except Exception as e:
            print(f"Warning: Failed to delete original annotations: {e}")

    sendProgress(
        1.0,
        "Complete", 
        f"Refined {len(new_annotations)} annotations from {total_annotations_processed} originals"
    )


if __name__ == '__main__':
    # Define the command-line interface for the entry point
    parser = argparse.ArgumentParser(
        description='SAM2 Refiner - Refine existing annotations using SAM2'
    )

    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str, required=True, action='store')

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
