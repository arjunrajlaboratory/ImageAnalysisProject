#!/usr/bin/env python3
"""
Local experiment script for SAM1 ViT-H few-shot segmentation.

Splits annotated data into train/test, runs the SAM1 pipeline,
and reports precision/recall/F1/IoU metrics. Optionally saves a
visualization image.

This is a direct adaptation of the SAM2 few-shot experiment script
to compare SAM1's ViT-H (2.6B params, trained on static images)
against SAM2's Hiera models for microscopy blob segmentation.

Usage:
    # Run experiment with defaults
    python test_sam1_fewshot_experiment.py --dataset DATASET_ID --tag "YFP blob"

    # Specify train/test split
    python test_sam1_fewshot_experiment.py --dataset DATASET_ID --tag "YFP blob" \
        --num-train 10 --seed 42

    # Save visualization
    python test_sam1_fewshot_experiment.py --dataset DATASET_ID --tag "YFP blob" \
        --visualize output.png

    # Interactive login
    python test_sam1_fewshot_experiment.py --dataset DATASET_ID --tag "YFP blob" \
        --username myuser
"""

import argparse
import getpass
import os
import sys
import time
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid
from skimage.measure import find_contours

import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import annotation_client.annotations as annotations_client
import annotation_client.tiles as tiles
import annotation_utilities.annotation_tools as annotation_tools


# ---------------------------------------------------------------------------
# Monkey-patch torchvision NMS to fix device mismatch bug
# (SAM1 + high points_per_side triggers a JIT-traced NMS path that has a
# CPU/GPU device mismatch; force the coordinate trick path which avoids it)
# ---------------------------------------------------------------------------
def _patch_torchvision_nms():
    try:
        import torchvision.ops.boxes as box_ops
        _coord_trick = box_ops._batched_nms_coordinate_trick

        def _safe_batched_nms(boxes, scores, idxs, iou_threshold):
            return _coord_trick(boxes, scores, idxs, iou_threshold)

        # Patch both the module-level function and SAM1's imported reference
        box_ops.batched_nms = _safe_batched_nms
        import segment_anything.automatic_mask_generator as sam_amg
        if hasattr(sam_amg, 'batched_nms'):
            sam_amg.batched_nms = _safe_batched_nms
    except Exception:
        pass

_patch_torchvision_nms()


# ---------------------------------------------------------------------------
# Utility functions (adapted from entrypoint.py for local use)
# ---------------------------------------------------------------------------

def ensure_rgb(image):
    """Ensure image is (H, W, 3) uint8 RGB."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype in (np.float32, np.float64):
        if image.max() <= 1.0 and image.min() >= 0.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return image


def extract_crop_with_context(image, mask, target_occupancy=0.20):
    """Crop image so the masked object occupies target_occupancy of area."""
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

    crop_area = obj_area / target_occupancy
    crop_side = int(np.sqrt(crop_area))
    crop_side = max(crop_side, obj_h, obj_w)

    cy = (y_min + y_max) / 2.0
    cx = (x_min + x_max) / 2.0
    h, w = image.shape[:2]

    half = crop_side / 2.0
    top = int(max(0, cy - half))
    left = int(max(0, cx - half))
    bottom = int(min(h, top + crop_side))
    right = int(min(w, left + crop_side))

    if bottom - top < crop_side and top > 0:
        top = max(0, bottom - crop_side)
    if right - left < crop_side and left > 0:
        left = max(0, right - crop_side)

    return image[top:bottom, left:right], mask[top:bottom, left:right]


def annotation_to_mask(annotation, image_shape):
    """Convert a polygon annotation to a binary mask."""
    from skimage.draw import polygon as draw_polygon

    coords = annotation['coordinates']
    rows = np.array([c['y'] for c in coords])
    cols = np.array([c['x'] for c in coords])

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    rr, cc = draw_polygon(rows, cols, shape=image_shape[:2])
    mask[rr, cc] = 1
    return mask


def annotation_to_shapely(annotation):
    """Convert annotation polygon to a Shapely Polygon."""
    coords = annotation['coordinates']
    pts = [(c['x'], c['y']) for c in coords]
    if len(pts) < 3:
        return None
    poly = ShapelyPolygon(pts)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.geom_type != 'Polygon':
        return None
    return poly


def encode_image_with_sam1(predictor, image_np):
    """Encode an image crop using SAM1's image encoder."""
    predictor.set_image(image_np)
    return predictor.features


def pool_features_with_mask(features, mask_np, feat_h, feat_w):
    """Pool feature map using a binary mask via weighted averaging."""
    mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_tensor, size=(feat_h, feat_w),
                                 mode='bilinear', align_corners=False)
    mask_resized = mask_resized.to(features.device)

    mask_sum = mask_resized.sum()
    if mask_sum > 0:
        weighted = (features * mask_resized).sum(dim=[2, 3]) / mask_sum
    else:
        weighted = features.mean(dim=[2, 3])

    return weighted.squeeze(0)


# ---------------------------------------------------------------------------
# Auth (same pattern as test_connection.py)
# ---------------------------------------------------------------------------

def get_auth(args):
    """Get API URL and token from env vars, CLI args, or interactive login."""
    api_url = args.api_url or os.environ.get('NIMBUS_API_URL', 'http://localhost:8080/api/v1')
    token = args.token or os.environ.get('NIMBUS_TOKEN')

    if token:
        print(f"API URL: {api_url}")
        print(f"Token:   {token[:8]}...")
        return api_url, token

    import girder_client
    gc = girder_client.GirderClient(apiUrl=api_url)
    username = args.username or os.environ.get('NIMBUS_USERNAME') or input("Username: ")
    password = args.password or os.environ.get('NIMBUS_PASSWORD') or getpass.getpass("Password: ")

    print(f"Logging in as '{username}' to {api_url}...")
    gc.authenticate(username, password)
    token = gc.token

    print(f"API URL: {api_url}")
    print(f"Token:   {token[:8]}...")
    return api_url, token


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_iou(poly_a, poly_b):
    """Compute Intersection over Union between two Shapely polygons."""
    if poly_a is None or poly_b is None:
        return 0.0
    try:
        inter = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def match_predictions_to_ground_truth(pred_polys, gt_polys, iou_threshold=0.3):
    """Match predicted polygons to ground-truth polygons using IoU.

    Uses greedy matching: for each GT polygon, find the best-matching
    prediction (highest IoU >= threshold). Each prediction can match at most
    one GT polygon.

    Returns:
        matches: list of (gt_idx, pred_idx, iou) tuples
        unmatched_gt: list of gt indices with no match
        unmatched_pred: list of pred indices with no match
    """
    if len(pred_polys) == 0 or len(gt_polys) == 0:
        return [], list(range(len(gt_polys))), list(range(len(pred_polys)))

    # Compute pairwise IoU
    iou_matrix = np.zeros((len(gt_polys), len(pred_polys)))
    for i, gt in enumerate(gt_polys):
        for j, pred in enumerate(pred_polys):
            iou_matrix[i, j] = compute_iou(gt, pred)

    matches = []
    matched_gt = set()
    matched_pred = set()

    # Greedy matching by highest IoU
    while True:
        if iou_matrix.size == 0:
            break
        best = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        best_iou = iou_matrix[best]
        if best_iou < iou_threshold:
            break

        gi, pi = best
        matches.append((gi, pi, best_iou))
        matched_gt.add(gi)
        matched_pred.add(pi)
        iou_matrix[gi, :] = -1
        iou_matrix[:, pi] = -1

    unmatched_gt = [i for i in range(len(gt_polys)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_polys)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


def compute_metrics(matches, num_gt, num_pred):
    """Compute precision, recall, F1 from matching results."""
    tp = len(matches)
    fp = num_pred - tp
    fn = num_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean([m[2] for m in matches]) if matches else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_results(image_rgb, gt_polys, pred_polys, matches,
                      unmatched_gt, unmatched_pred, metrics, output_path):
    """Save a visualization showing GT vs predicted annotations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: Ground truth
    axes[0].imshow(image_rgb)
    axes[0].set_title(f'Ground Truth ({len(gt_polys)} objects)')
    for poly in gt_polys:
        if poly is not None:
            xs, ys = poly.exterior.xy
            axes[0].plot(xs, ys, 'g-', linewidth=1.5)

    # Panel 2: Predictions
    axes[1].imshow(image_rgb)
    axes[1].set_title(f'Predictions ({len(pred_polys)} objects)')
    for poly in pred_polys:
        if poly is not None:
            xs, ys = poly.exterior.xy
            axes[1].plot(xs, ys, 'r-', linewidth=1.5)

    # Panel 3: Overlay with match coloring
    axes[2].imshow(image_rgb)
    axes[2].set_title(f'Overlay (P={metrics["precision"]:.2f} R={metrics["recall"]:.2f} F1={metrics["f1"]:.2f})')

    # Matched GT in green, matched pred in blue
    for gi, pi, iou in matches:
        gt = gt_polys[gi]
        pred = pred_polys[pi]
        if gt is not None:
            xs, ys = gt.exterior.xy
            axes[2].plot(xs, ys, 'g-', linewidth=1.5, alpha=0.8)
        if pred is not None:
            xs, ys = pred.exterior.xy
            axes[2].plot(xs, ys, 'b--', linewidth=1.5, alpha=0.8)

    # Unmatched GT in yellow (false negatives)
    for gi in unmatched_gt:
        gt = gt_polys[gi]
        if gt is not None:
            xs, ys = gt.exterior.xy
            axes[2].plot(xs, ys, 'y-', linewidth=2, alpha=0.9)

    # Unmatched pred in red (false positives)
    for pi in unmatched_pred:
        pred = pred_polys[pi]
        if pred is not None:
            xs, ys = pred.exterior.xy
            axes[2].plot(xs, ys, 'r-', linewidth=2, alpha=0.9)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to {output_path}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    api_url, token = get_auth(args)
    dataset_id = args.dataset

    tag = args.tag
    num_train = args.num_train
    seed = args.seed
    similarity_threshold = args.threshold
    target_occupancy = args.occupancy
    points_per_side = args.points_per_side
    min_mask_area = args.min_area
    max_mask_area = args.max_area
    smoothing = args.smoothing
    iou_threshold = args.iou_threshold

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = args.checkpoint_dir
    model_name = args.model

    print("\n" + "=" * 60)
    print("SAM1 VIT-H FEW-SHOT SEGMENTATION EXPERIMENT")
    print("=" * 60)
    print(f"  Dataset:     {dataset_id}")
    print(f"  Tag:         {tag}")
    print(f"  Num train:   {num_train}")
    print(f"  Seed:        {seed}")
    print(f"  Threshold:   {similarity_threshold}")
    print(f"  Occupancy:   {target_occupancy}")
    print(f"  Pts/side:    {points_per_side}")
    print(f"  Min area:    {min_mask_area}")
    print(f"  Max area:    {max_mask_area}")
    print(f"  Smoothing:   {smoothing}")
    print(f"  IoU thresh:  {iou_threshold}")
    print(f"  Model:       {model_name}")
    print(f"  Device:      {device}")

    # ── Fetch annotations ──
    print("\n--- Fetching annotations ---")
    annotationClient = annotations_client.UPennContrastAnnotationClient(
        apiUrl=api_url, token=token
    )
    tileClient = tiles.UPennContrastDataset(
        apiUrl=api_url, token=token, datasetId=dataset_id
    )

    all_polygons = annotationClient.getAnnotationsByDatasetId(dataset_id, shape='polygon')
    tagged = annotation_tools.get_annotations_with_tags(all_polygons, [tag], exclusive=False)
    print(f"  Total polygons in dataset: {len(all_polygons)}")
    print(f"  Polygons with tag '{tag}': {len(tagged)}")

    if len(tagged) < 3:
        print("ERROR: Need at least 3 annotations to split into train/test.")
        sys.exit(1)

    # ── Group by location ──
    by_location = defaultdict(list)
    for ann in tagged:
        loc = ann.get('location', {})
        key = (loc.get('XY', 0), loc.get('Z', 0), loc.get('Time', 0))
        by_location[key].append(ann)

    print(f"  Annotations span {len(by_location)} location(s):")
    for loc, anns in by_location.items():
        print(f"    XY={loc[0]} Z={loc[1]} T={loc[2]}: {len(anns)} annotations")

    # ── Train/test split ──
    rng = np.random.RandomState(seed)

    if num_train >= len(tagged):
        print(f"WARNING: num_train ({num_train}) >= total ({len(tagged)}), using all but 1 for train")
        num_train = len(tagged) - 1

    indices = rng.permutation(len(tagged))
    train_indices = set(indices[:num_train])

    train_annotations = [tagged[i] for i in range(len(tagged)) if i in train_indices]
    test_annotations = [tagged[i] for i in range(len(tagged)) if i not in train_indices]

    print(f"\n  Train: {len(train_annotations)}, Test: {len(test_annotations)}")

    # ── Load SAM1 ViT-H ──
    print("\n--- Loading SAM1 ViT-H ---")

    checkpoint_path = os.path.join(checkpoint_dir, model_name)
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    # Enable CUDA optimizations (no autocast for SAM1 — its mask generator
    # calls .numpy() on intermediate tensors which is incompatible with bfloat16)
    if device == 'cuda':
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    sam_model = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam_model.to(device)
    predictor = SamPredictor(sam_model)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ── Phase 1: Extract training prototype ──
    print("\n--- Phase 1: Training feature extraction ---")
    feature_vectors = []

    for idx, annotation in enumerate(train_annotations):
        loc = annotation['location']
        ann_xy = loc.get('XY', 0)
        ann_z = loc.get('Z', 0)
        ann_time = loc.get('Time', 0)

        images = annotation_tools.get_images_for_all_channels(
            tileClient, dataset_id, ann_xy, ann_z, ann_time)
        layers = annotation_tools.get_layers(tileClient.client, dataset_id)
        merged_image = annotation_tools.process_and_merge_channels(images, layers)
        merged_image = ensure_rgb(merged_image)

        mask = annotation_to_mask(annotation, merged_image.shape)
        if mask.sum() == 0:
            print(f"  WARNING: annotation {idx} produced empty mask, skipping")
            continue

        crop_image, crop_mask = extract_crop_with_context(merged_image, mask, target_occupancy)
        crop_image = ensure_rgb(crop_image)

        features = encode_image_with_sam1(predictor, crop_image)
        feat_h, feat_w = features.shape[2], features.shape[3]
        feature_vec = pool_features_with_mask(features, crop_mask, feat_h, feat_w)
        feature_vectors.append(feature_vec)

        print(f"  [{idx + 1}/{len(train_annotations)}] Encoded training example "
              f"(area={mask.sum()}, crop={crop_image.shape})")

    if len(feature_vectors) == 0:
        print("ERROR: No valid training features extracted.")
        sys.exit(1)

    training_prototype = torch.stack(feature_vectors).mean(dim=0)
    training_prototype = F.normalize(training_prototype.unsqueeze(0), dim=1).squeeze(0)
    print(f"  Prototype: shape={training_prototype.shape}, "
          f"from {len(feature_vectors)} vectors")

    # ── Phase 2: Inference on test locations ──
    print("\n--- Phase 2: Inference ---")

    test_by_location = defaultdict(list)
    for ann in test_annotations:
        loc = ann.get('location', {})
        key = (loc.get('XY', 0), loc.get('Z', 0), loc.get('Time', 0))
        test_by_location[key].append(ann)

    test_locations = list(test_by_location.keys())
    print(f"  Test locations: {len(test_locations)}")

    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=min_mask_area,
    )

    all_gt_polys = []
    all_pred_polys = []
    all_images = {}

    for loc_idx, (xy, z, t) in enumerate(test_locations):
        print(f"\n  Location {loc_idx + 1}/{len(test_locations)}: XY={xy} Z={z} T={t}")

        # Load merged image
        images = annotation_tools.get_images_for_all_channels(
            tileClient, dataset_id, xy, z, t)
        layers = annotation_tools.get_layers(tileClient.client, dataset_id)
        merged = annotation_tools.process_and_merge_channels(images, layers)
        merged_rgb = ensure_rgb(merged)
        all_images[(xy, z, t)] = merged_rgb

        # Ground truth polygons for this location
        gt_annotations = test_by_location[(xy, z, t)]
        gt_polys = [annotation_to_shapely(a) for a in gt_annotations]
        gt_polys = [p for p in gt_polys if p is not None]

        print(f"    GT polygons: {len(gt_polys)}")

        # Generate candidate masks
        t0 = time.time()
        candidates = mask_generator.generate(merged_rgb)
        print(f"    Candidates: {len(candidates)} (generated in {time.time() - t0:.1f}s)")

        # Filter by similarity
        pred_polys = []
        similarities = []
        t0 = time.time()

        for ci, mask_data in enumerate(candidates):
            mask = mask_data['segmentation']
            area = mask.sum()

            if min_mask_area > 0 and area < min_mask_area:
                continue
            if max_mask_area > 0 and area > max_mask_area:
                continue

            crop_image, crop_mask = extract_crop_with_context(
                merged_rgb, mask, target_occupancy)
            crop_image = ensure_rgb(crop_image)

            if crop_mask.sum() == 0:
                continue

            features = encode_image_with_sam1(predictor, crop_image)
            feat_h, feat_w = features.shape[2], features.shape[3]
            feature_vec = pool_features_with_mask(features, crop_mask, feat_h, feat_w)

            feature_vec_norm = F.normalize(feature_vec.unsqueeze(0), dim=1)
            similarity = F.cosine_similarity(
                feature_vec_norm, training_prototype.unsqueeze(0)
            ).item()
            similarities.append(similarity)

            if similarity >= similarity_threshold:
                contours = find_contours(mask, 0.5)
                if len(contours) == 0:
                    continue
                # find_contours returns (row, col) = (y, x)
                poly = ShapelyPolygon([(c[1], c[0]) for c in contours[0]])
                poly = poly.simplify(smoothing, preserve_topology=True)
                if poly.is_valid and not poly.is_empty and poly.geom_type == 'Polygon':
                    pred_polys.append(poly)

        print(f"    Encoding took {time.time() - t0:.1f}s")
        if similarities:
            print(f"    Similarity stats: min={min(similarities):.3f}, "
                  f"max={max(similarities):.3f}, mean={np.mean(similarities):.3f}")
        print(f"    Predictions passing threshold: {len(pred_polys)}")

        all_gt_polys.extend(gt_polys)
        all_pred_polys.extend(pred_polys)

        # Per-location metrics
        matches, unmatched_gt, unmatched_pred = match_predictions_to_ground_truth(
            pred_polys, gt_polys, iou_threshold
        )
        loc_metrics = compute_metrics(matches, len(gt_polys), len(pred_polys))
        print(f"    P={loc_metrics['precision']:.3f} R={loc_metrics['recall']:.3f} "
              f"F1={loc_metrics['f1']:.3f} IoU={loc_metrics['mean_iou']:.3f} "
              f"(TP={loc_metrics['tp']} FP={loc_metrics['fp']} FN={loc_metrics['fn']})")

    # ── Overall metrics ──
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    matches, unmatched_gt, unmatched_pred = match_predictions_to_ground_truth(
        all_pred_polys, all_gt_polys, iou_threshold
    )
    overall = compute_metrics(matches, len(all_gt_polys), len(all_pred_polys))

    print(f"  Ground truth:  {len(all_gt_polys)}")
    print(f"  Predictions:   {len(all_pred_polys)}")
    print(f"  True positives:  {overall['tp']}")
    print(f"  False positives: {overall['fp']}")
    print(f"  False negatives: {overall['fn']}")
    print(f"  Precision:     {overall['precision']:.4f}")
    print(f"  Recall:        {overall['recall']:.4f}")
    print(f"  F1 Score:      {overall['f1']:.4f}")
    print(f"  Mean IoU:      {overall['mean_iou']:.4f}")

    # ── Visualization ──
    if args.visualize and len(all_images) > 0:
        print(f"\n--- Saving visualization ---")
        first_loc = test_locations[0]
        img = all_images[first_loc]

        gt_anns = test_by_location[first_loc]
        gt_for_viz = [annotation_to_shapely(a) for a in gt_anns]
        gt_for_viz = [p for p in gt_for_viz if p is not None]

        viz_matches, viz_unmatched_gt, viz_unmatched_pred = match_predictions_to_ground_truth(
            all_pred_polys[:len(test_by_location[first_loc])],
            gt_for_viz,
            iou_threshold
        )
        viz_metrics = compute_metrics(viz_matches, len(gt_for_viz),
                                      len(all_pred_polys[:len(test_by_location[first_loc])]))

        visualize_results(img, gt_for_viz,
                          all_pred_polys[:len(test_by_location[first_loc])],
                          viz_matches, viz_unmatched_gt, viz_unmatched_pred,
                          viz_metrics, args.visualize)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='SAM1 ViT-H few-shot segmentation experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset ID')
    parser.add_argument('--tag', type=str, required=True,
                        help='Annotation tag to use for train/test')

    # Auth
    parser.add_argument('--api-url', type=str, default=None,
                        help='API URL (default: $NIMBUS_API_URL)')
    parser.add_argument('--token', type=str, default=None,
                        help='Auth token (default: $NIMBUS_TOKEN)')
    parser.add_argument('--username', type=str, default=None)
    parser.add_argument('--password', type=str, default=None)

    # Experiment
    parser.add_argument('--num-train', type=int, default=10,
                        help='Number of training annotations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split')

    # SAM1 parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt_dir = os.path.join(script_dir, 'checkpoints')

    parser.add_argument('--model', type=str, default='sam_vit_h_4b8939.pth',
                        help='SAM1 checkpoint name')
    parser.add_argument('--checkpoint-dir', type=str,
                        default=default_ckpt_dir,
                        help='Directory containing SAM1 checkpoints')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Similarity threshold')
    parser.add_argument('--occupancy', type=float, default=0.20,
                        help='Target occupancy for cropping')
    parser.add_argument('--points-per-side', type=int, default=32,
                        help='Points per side for mask generation')
    parser.add_argument('--min-area', type=int, default=100,
                        help='Minimum mask area in pixels')
    parser.add_argument('--max-area', type=int, default=0,
                        help='Maximum mask area (0 = no limit)')
    parser.add_argument('--smoothing', type=float, default=0.3,
                        help='Polygon simplification tolerance')

    # Evaluation
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='IoU threshold for matching predictions to GT')

    # Output
    parser.add_argument('--visualize', type=str, default=None,
                        help='Path to save visualization image (e.g., output.png)')

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
