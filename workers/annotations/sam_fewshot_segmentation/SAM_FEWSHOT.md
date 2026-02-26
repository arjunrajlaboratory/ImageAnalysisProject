# SAM Few-Shot Segmentation Worker

## Overview

This worker segments objects in microscopy images using few-shot learning with SAM1 ViT-H (2.6B parameters). Users annotate a small number of training examples (5-20 objects) with a specific tag, and the worker finds similar objects across the dataset using SAM1's frozen image encoder features. No model training is required.

This is the SAM1 counterpart to the SAM2 few-shot segmentation worker. SAM1 ViT-H produces higher quality masks (F1=0.62 vs 0.53 for SAM2 base_plus at 256 pts/side in our experiments) but is slower (~8x slower encoding per image).

## How It Works

### Phase 1: Training Feature Extraction

For each polygon annotation matching the user-specified Training Tag:

1. Load the merged multi-channel image at the annotation's location
2. Convert the annotation polygon to a binary mask
3. Crop the image around the object with context padding (object occupies ~20% of crop area by default)
4. Encode the crop through SAM1's image encoder via `SamPredictor.set_image()`
5. Extract the `features` tensor (shape: `1, 256, 64, 64` for ViT-H)
6. Pool the feature map using mask-weighted averaging to produce a 256-dimensional feature vector
7. Average all training feature vectors into a single L2-normalized prototype

### Phase 2: Inference

For each image frame in the batch:

1. Run `SamAutomaticMaskGenerator` to generate all candidate masks
2. For each candidate mask:
   - Apply the same crop-encode-pool pipeline as training
   - Compute cosine similarity between the candidate's feature vector and the training prototype
   - Keep the mask if similarity >= threshold
3. Convert passing masks to polygon annotations via `find_contours` + `polygons_to_annotations`
4. Upload all annotations to the server

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Training Tag | tags | (required) | Tag identifying training annotation examples |
| Batch XY | text | current | XY positions to process (e.g., "1-3, 5-8") |
| Batch Z | text | current | Z slices to process |
| Batch Time | text | current | Time points to process |
| Model | select | sam_vit_h_4b8939 | SAM1 checkpoint (ViT-H only) |
| Similarity Threshold | number | 0.5 | Minimum cosine similarity to keep a mask (0.0-1.0) |
| Target Occupancy | number | 0.20 | Fraction of crop area the object should occupy (0.05-0.80) |
| Points per side | number | 128 | Grid density for SAM1 mask generation (16-128) |
| Min Mask Area | number | 30 | Minimum mask area in pixels to consider |
| Max Mask Area | number | 0 | Maximum mask area in pixels (0 = no limit) |
| Smoothing | number | 0.3 | Polygon simplification tolerance |

## Key Design Decisions

### SAM1 vs SAM2

SAM1 ViT-H was trained on static images (SA-1B dataset, 11M images) and has a larger encoder (2.6B params) compared to SAM2's Hiera models. In our microscopy experiments:

- **SAM1 ViT-H @ 128 pts/side**: F1=0.62, Recall=0.57, Precision=0.70
- **SAM2 base_plus @ 256 pts/side**: F1=0.53, Recall=0.40, Precision=0.77

SAM1 is the better choice when mask quality matters more than speed.

### No bfloat16 Autocast

Unlike the SAM2 worker, this worker does NOT use `torch.autocast(dtype=torch.bfloat16)`. SAM1's `SamAutomaticMaskGenerator` calls `.numpy()` on intermediate tensors during mask generation, which is incompatible with bfloat16 autocast. TF32 is still enabled on Ampere+ GPUs for acceleration.

### NMS Monkey-Patch

At high `points_per_side` values (128+), SAM1's mask generator triggers a JIT-traced NMS code path in torchvision that has a CPU/GPU device mismatch bug. The worker monkey-patches `torchvision.ops.boxes.batched_nms` to always use the coordinate trick path, which avoids this issue.

### Points per side capped at 128

The maximum `points_per_side` is capped at 128 (vs 256 for SAM2) because values above 128 can trigger additional NMS issues beyond what the monkey-patch covers. At 128 pts/side, SAM1 already generates sufficient candidates for most microscopy applications.

### Feature Access

SAM1 uses `predictor.features` (a direct tensor attribute) vs SAM2's `predictor._features["image_embed"]` (a dict lookup). Both produce the same semantic feature maps from their respective encoders.

### Context Padding (Target Occupancy)

Same as SAM2 worker. SAM models were trained on images where objects occupy a reasonable fraction of the frame. The `Target Occupancy` parameter controls how much of the crop the object fills (default 20%).

### Mask-Weighted Feature Pooling

Same as SAM2 worker. Binary masks focus feature pooling on actual object pixels rather than background.

## Tuning Guide

### Similarity Threshold

- **Too many false positives**: Increase threshold (try 0.6-0.8)
- **Too few detections (missing objects)**: Decrease threshold (try 0.3-0.4)
- **Start at 0.5** and adjust based on results

### Target Occupancy

- **Objects are very small in the image**: Try 0.10-0.15 (more context)
- **Objects are large in the image**: Try 0.30-0.40 (less context)
- **Default 0.20** works well for most microscopy objects

### Points per side

- **More masks needed (small objects)**: Use the maximum 128
- **Faster processing**: Decrease to 32-64
- **Default 128** provides best recall at the cost of longer mask generation

### Min/Max Mask Area

- Use training annotation areas as a guide
- Set Min to ~50% of smallest training annotation area
- Set Max to ~200% of largest training annotation area
- Set Max to 0 to disable upper limit

## Performance Characteristics

- **GPU required**: SAM1 ViT-H encoder needs CUDA
- **Memory**: ~8GB VRAM for SAM1 ViT-H model
- **Speed**: SAM1 ViT-H encoding is ~8x slower per image than SAM2 Hiera models. With 128 points per side, expect ~200-800 candidate masks per image.
- **Data efficiency**: Works with 5-20 training examples
- **Quality**: Higher F1 and recall than SAM2 in microscopy experiments

## Files

| File | Purpose |
|------|---------|
| `entrypoint.py` | Worker logic: interface definition, feature extraction, inference pipeline |
| `Dockerfile` | x86_64 production build (CUDA 11.8, SAM1 ViT-H checkpoint) |
| `Dockerfile_M1` | arm64/M1 Mac build (CUDA 11.8) |
| `environment.yml` | Conda environment specification |
| `tests/test_sam_fewshot.py` | Unit tests for helper functions |
| `tests/Dockerfile_Test` | Test Docker image |
