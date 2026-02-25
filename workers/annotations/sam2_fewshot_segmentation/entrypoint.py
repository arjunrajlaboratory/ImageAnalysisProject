import argparse
import json
import sys
import os

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
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor

from annotation_client.utils import sendProgress, sendError


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    models = [f for f in os.listdir('/code/sam2/checkpoints') if f.endswith('.pt')]
    default_model = 'sam2.1_hiera_small.pt' if 'sam2.1_hiera_small.pt' in models else models[0] if models else None

    interface = {
        'Training Tag': {
            'type': 'tags',
            'displayOrder': 0,
        },
        'Batch XY': {
            'type': 'text',
            'displayOrder': 1,
        },
        'Batch Z': {
            'type': 'text',
            'displayOrder': 2,
        },
        'Batch Time': {
            'type': 'text',
            'displayOrder': 3,
        },
        'Model': {
            'type': 'select',
            'items': models,
            'default': default_model,
            'displayOrder': 4,
        },
        'Similarity Threshold': {
            'type': 'number',
            'min': 0.0,
            'max': 1.0,
            'default': 0.5,
            'displayOrder': 5,
        },
        'Target Occupancy': {
            'type': 'number',
            'min': 0.05,
            'max': 0.80,
            'default': 0.20,
            'displayOrder': 6,
        },
        'Points per side': {
            'type': 'number',
            'min': 16,
            'max': 128,
            'default': 32,
            'displayOrder': 7,
        },
        'Min Mask Area': {
            'type': 'number',
            'min': 0,
            'max': 100000,
            'default': 100,
            'displayOrder': 8,
        },
        'Max Mask Area': {
            'type': 'number',
            'min': 0,
            'max': 10000000,
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
    client.setWorkerImageInterface(image, interface)


def extract_crop_with_context(image, mask, target_occupancy=0.20):
    """Extract a crop of the image where the masked object occupies roughly
    target_occupancy fraction of the crop area.

    Args:
        image: numpy array (H, W, C) or (H, W)
        mask: binary numpy array (H, W)
        target_occupancy: desired fraction of crop area occupied by object

    Returns:
        crop_image: numpy array resized/cropped region
        crop_mask: binary numpy array of same spatial size as crop_image
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return image, mask

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    obj_h = y_max - y_min + 1
    obj_w = x_max - x_min + 1

    obj_area = mask.sum()
    if obj_area == 0:
        return image, mask

    # Determine crop size so that object occupies target_occupancy of area
    crop_area = obj_area / target_occupancy
    crop_side = int(np.sqrt(crop_area))
    # Ensure crop is at least as large as the object bounding box
    crop_side = max(crop_side, obj_h, obj_w)

    # Center the crop on the object center
    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0

    h, w = image.shape[:2]

    half = crop_side / 2.0
    top = int(max(0, cy - half))
    left = int(max(0, cx - half))
    bottom = int(min(h, top + crop_side))
    right = int(min(w, left + crop_side))

    # Adjust if we hit boundaries
    if bottom - top < crop_side and top > 0:
        top = max(0, bottom - crop_side)
    if right - left < crop_side and left > 0:
        left = max(0, right - crop_side)

    crop_image = image[top:bottom, left:right]
    crop_mask = mask[top:bottom, left:right]

    return crop_image, crop_mask


def encode_image_with_sam2(predictor, image_np):
    """Encode an image crop using SAM2's image encoder via SAM2ImagePredictor.

    Uses set_image() which handles transforms, backbone encoding, and
    no_mem_embed addition consistently with SAM2's internal pipeline.

    Args:
        predictor: SAM2ImagePredictor instance
        image_np: numpy array (H, W, 3) uint8

    Returns:
        features: tensor of shape [1, 256, 64, 64] (image_embed)
    """
    predictor.set_image(image_np)
    # image_embed is the lowest-resolution, highest-semantic feature map
    # Shape: (1, 256, 64, 64) for 1024x1024 input
    return predictor._features["image_embed"]


def pool_features_with_mask(features, mask_np, feat_h, feat_w):
    """Pool feature map using a binary mask via weighted averaging.

    Args:
        features: tensor (1, C, feat_h, feat_w)
        mask_np: binary numpy array (crop_h, crop_w)
        feat_h: feature map height
        feat_w: feature map width

    Returns:
        feature_vector: tensor of shape (C,)
    """
    # Resize mask to feature map dimensions
    mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_tensor, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
    mask_resized = mask_resized.to(features.device)

    # Weighted pooling
    mask_sum = mask_resized.sum()
    if mask_sum > 0:
        weighted = (features * mask_resized).sum(dim=[2, 3]) / mask_sum
    else:
        weighted = features.mean(dim=[2, 3])

    return weighted.squeeze(0)  # (C,)


def ensure_rgb(image):
    """Ensure image is (H, W, 3) uint8 RGB."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0 and image.min() >= 0.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return image


def annotation_to_mask(annotation, image_shape):
    """Convert a polygon annotation to a binary mask.

    Args:
        annotation: annotation dict with 'coordinates' list of {'x': ..., 'y': ...}
        image_shape: (H, W) of the target image

    Returns:
        mask: binary numpy array (H, W)
    """
    from skimage.draw import polygon as draw_polygon

    coords = annotation['coordinates']
    # Annotation coordinates: 'x' and 'y' in image pixel space
    rows = np.array([c['y'] for c in coords])
    cols = np.array([c['x'] for c in coords])

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    rr, cc = draw_polygon(rows, cols, shape=image_shape[:2])
    mask[rr, cc] = 1
    return mask


def compute(datasetId, apiUrl, token, params):
    annotationClient = annotations_client.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    # Parse parameters
    model_name = params['workerInterface']['Model']
    similarity_threshold = float(params['workerInterface']['Similarity Threshold'])
    target_occupancy = float(params['workerInterface']['Target Occupancy'])
    points_per_side = int(params['workerInterface']['Points per side'])
    min_mask_area = int(params['workerInterface']['Min Mask Area'])
    max_mask_area = int(params['workerInterface']['Max Mask Area'])
    smoothing = float(params['workerInterface']['Smoothing'])

    batch_xy = params['workerInterface'].get('Batch XY', '')
    batch_z = params['workerInterface'].get('Batch Z', '')
    batch_time = params['workerInterface'].get('Batch Time', '')

    batch_xy = batch_argument_parser.process_range_list(batch_xy, convert_one_to_zero_index=True)
    batch_z = batch_argument_parser.process_range_list(batch_z, convert_one_to_zero_index=True)
    batch_time = batch_argument_parser.process_range_list(batch_time, convert_one_to_zero_index=True)

    tile = params['tile']
    channel = params['channel']
    tags = params['tags']

    if batch_xy is None:
        batch_xy = [tile['XY']]
    if batch_z is None:
        batch_z = [tile['Z']]
    if batch_time is None:
        batch_time = [tile['Time']]

    # Parse training tag - 'type': 'tags' returns a list of strings directly
    training_tags = params['workerInterface'].get('Training Tag', [])
    if not training_tags or len(training_tags) == 0:
        sendError("No training tag selected",
                  "Please select a tag that identifies your training annotations.")
        return

    # ── SAM2 model setup ──
    sendProgress(0.0, "Loading model", "Initializing SAM2...")
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_path = f"/code/sam2/checkpoints/{model_name}"
    model_to_cfg = {
        'sam2.1_hiera_base_plus.pt': 'sam2.1_hiera_b+.yaml',
        'sam2.1_hiera_large.pt': 'sam2.1_hiera_l.yaml',
        'sam2.1_hiera_small.pt': 'sam2.1_hiera_s.yaml',
        'sam2.1_hiera_tiny.pt': 'sam2.1_hiera_t.yaml',
    }
    model_cfg = f"configs/sam2.1/{model_to_cfg[model_name]}"
    sam2_model = build_sam2(model_cfg, checkpoint_path, device='cuda', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2_model)

    # ── Phase 1: Extract training prototype ──
    sendProgress(0.05, "Extracting training features", "Fetching training annotations...")

    # Fetch all polygon annotations from the dataset
    all_annotations = annotationClient.getAnnotationsByDatasetId(datasetId, shape='polygon')
    training_annotations = annotation_tools.get_annotations_with_tags(
        all_annotations, training_tags, exclusive=False
    )

    if len(training_annotations) == 0:
        sendError("No training annotations found", f"No polygon annotations found with tags: {training_tags}")
        return

    print(f"Found {len(training_annotations)} training annotations")

    feature_vectors = []
    for idx, annotation in enumerate(training_annotations):
        loc = annotation['location']
        ann_xy = loc.get('XY', 0)
        ann_z = loc.get('Z', 0)
        ann_time = loc.get('Time', 0)

        # Get the merged image at the annotation's location
        images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, ann_xy, ann_z, ann_time)
        layers = annotation_tools.get_layers(tileClient.client, datasetId)
        merged_image = annotation_tools.process_and_merge_channels(images, layers)
        merged_image = ensure_rgb(merged_image)

        # Convert annotation to mask
        mask = annotation_to_mask(annotation, merged_image.shape)

        if mask.sum() == 0:
            print(f"Warning: training annotation {idx} produced empty mask, skipping")
            continue

        # Extract crop with context padding
        crop_image, crop_mask = extract_crop_with_context(merged_image, mask, target_occupancy)
        crop_image = ensure_rgb(crop_image)

        # Encode with SAM2
        features = encode_image_with_sam2(predictor, crop_image)
        feat_h, feat_w = features.shape[2], features.shape[3]

        # Pool features with mask
        feature_vec = pool_features_with_mask(features, crop_mask, feat_h, feat_w)
        feature_vectors.append(feature_vec)

        sendProgress(0.05 + 0.15 * (idx + 1) / len(training_annotations),
                     "Extracting training features",
                     f"Processed {idx + 1}/{len(training_annotations)} training examples")

    if len(feature_vectors) == 0:
        sendError("No valid training features", "All training annotations produced empty masks")
        return

    # Create prototype by averaging feature vectors
    training_prototype = torch.stack(feature_vectors).mean(dim=0)
    training_prototype = F.normalize(training_prototype.unsqueeze(0), dim=1).squeeze(0)

    print(f"Training prototype shape: {training_prototype.shape}")

    # Optionally learn size statistics from training annotations
    training_areas = []
    for annotation in training_annotations:
        coords = annotation['coordinates']
        rows = [c['y'] for c in coords]
        cols = [c['x'] for c in coords]
        poly = Polygon(zip(cols, rows))
        if poly.is_valid:
            training_areas.append(poly.area)

    mean_area = np.mean(training_areas) if training_areas else None
    std_area = np.std(training_areas) if training_areas else None
    print(f"Training area stats: mean={mean_area}, std={std_area}")

    # ── Phase 2: Inference ──
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=min_mask_area,
    )

    batches = list(product(batch_xy, batch_z, batch_time))
    total_batches = len(batches)
    new_annotations = []

    for i, batch in enumerate(batches):
        XY, Z, Time = batch

        sendProgress(0.2 + 0.7 * i / total_batches,
                     "Segmenting",
                     f"Processing frame {i + 1}/{total_batches}")

        # Get merged image for this batch
        images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, XY, Z, Time)
        layers = annotation_tools.get_layers(tileClient.client, datasetId)
        merged_image = annotation_tools.process_and_merge_channels(images, layers)
        merged_image_rgb = ensure_rgb(merged_image)

        # Generate candidate masks with SAM2
        candidate_masks = mask_generator.generate(merged_image_rgb)
        print(f"Frame {i + 1}: generated {len(candidate_masks)} candidate masks")

        # Filter candidates by similarity to training prototype
        filtered_polygons = []
        for mask_data in candidate_masks:
            mask = mask_data['segmentation']
            area = mask.sum()

            # Area filtering
            if min_mask_area > 0 and area < min_mask_area:
                continue
            if max_mask_area > 0 and area > max_mask_area:
                continue

            # Extract crop with context, encode, and compare
            crop_image, crop_mask = extract_crop_with_context(
                merged_image_rgb, mask, target_occupancy
            )
            crop_image = ensure_rgb(crop_image)

            if crop_mask.sum() == 0:
                continue

            features = encode_image_with_sam2(predictor, crop_image)
            feat_h, feat_w = features.shape[2], features.shape[3]
            feature_vec = pool_features_with_mask(features, crop_mask, feat_h, feat_w)

            # Compute cosine similarity
            feature_vec_norm = F.normalize(feature_vec.unsqueeze(0), dim=1)
            similarity = F.cosine_similarity(
                feature_vec_norm,
                training_prototype.unsqueeze(0)
            ).item()

            if similarity >= similarity_threshold:
                # Convert mask to polygon
                contours = find_contours(mask, 0.5)
                if len(contours) == 0:
                    continue
                polygon = Polygon(contours[0]).simplify(smoothing, preserve_topology=True)
                if polygon.is_valid and not polygon.is_empty:
                    filtered_polygons.append(polygon)

        print(f"Frame {i + 1}: {len(filtered_polygons)} masks passed similarity filter")

        # Convert polygons to annotations
        temp_annotations = annotation_tools.polygons_to_annotations(
            filtered_polygons, datasetId, XY=XY, Time=Time, Z=Z, tags=tags, channel=channel
        )
        new_annotations.extend(temp_annotations)

    sendProgress(0.9, "Uploading annotations", f"Sending {len(new_annotations)} annotations to server")
    annotationClient.createMultipleAnnotations(new_annotations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SAM2 Few-Shot Segmentation')

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
