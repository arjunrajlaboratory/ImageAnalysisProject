# SAM2 Few-Shot Segmentation Worker

## Overview

This worker segments objects in microscopy images using few-shot learning with SAM2. Users annotate a small number of training examples (5-20 objects) with a specific tag, and the worker finds similar objects across the dataset using SAM2's frozen image encoder features. No model training is required.

## How It Works

### Phase 1: Training Feature Extraction

For each polygon annotation matching the user-specified Training Tag:

1. Load the merged multi-channel image at the annotation's location
2. Convert the annotation polygon to a binary mask
3. Crop the image around the object with context padding (object occupies ~20% of crop area by default)
4. Encode the crop through SAM2's image encoder via `SAM2ImagePredictor.set_image()`
5. Extract the `image_embed` feature map (shape: `1, 256, 64, 64`)
6. Pool the feature map using mask-weighted averaging to produce a 256-dimensional feature vector
7. Average all training feature vectors into a single L2-normalized prototype

### Phase 2: Inference

For each image frame in the batch:

1. Run `SAM2AutomaticMaskGenerator` to generate all candidate masks
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
| Model | select | sam2.1_hiera_base_plus.pt | SAM2 checkpoint to use |
| Similarity Threshold | number | 0.5 | Minimum cosine similarity to keep a mask (0.0-1.0) |
| Target Occupancy | number | 0.20 | Fraction of crop area the object should occupy (0.05-0.80) |
| Points per side | number | 128 | Grid density for SAM2 mask generation (16-256) |
| Min Mask Area | number | 30 | Minimum mask area in pixels to consider |
| Max Mask Area | number | 0 | Maximum mask area in pixels (0 = no limit) |
| Smoothing | number | 0.3 | Polygon simplification tolerance |

## Key Design Decisions

### Context Padding (Target Occupancy)

SAM2 was trained on images where objects occupy a reasonable fraction of the frame. Tight crops around objects would be out-of-distribution. The `Target Occupancy` parameter controls how much of the crop the object fills:

- `crop_side = sqrt(object_area / target_occupancy)`
- Default 0.20 means the object occupies ~20% of the crop area
- The same occupancy is used for both training and inference to ensure consistent feature extraction

### Mask-Weighted Feature Pooling

Since we have binary masks for both training annotations and candidate masks, we use them to focus the feature pooling on the actual object pixels rather than background:

```
feature_vector = (features * mask).sum(dim=[2,3]) / mask.sum()
```

The mask is bilinearly resized from the crop resolution to the feature map resolution (64x64).

### SAM2ImagePredictor for Encoding

We use `SAM2ImagePredictor.set_image()` rather than calling `forward_image` directly. This ensures proper handling of:
- Image transforms (resize to 1024x1024, normalization)
- `no_mem_embed` addition (SAM2's learned "no memory" token)
- Consistent feature extraction matching SAM2's internal pipeline

The `image_embed` from `predictor._features["image_embed"]` gives a `(1, 256, 64, 64)` feature map -- the lowest-resolution, highest-semantic features from SAM2's FPN neck.

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

- **More masks needed (small objects)**: Increase to 192-256
- **Faster processing**: Decrease to 32-64
- **Default 128** provides good coverage for most microscopy images

### Min/Max Mask Area

- Use training annotation areas as a guide
- Set Min to ~50% of smallest training annotation area
- Set Max to ~200% of largest training annotation area
- Set Max to 0 to disable upper limit

## Performance Characteristics

- **GPU required**: SAM2 encoder needs CUDA
- **Memory**: ~4GB VRAM for SAM2 small model
- **Speed**: Most time is spent encoding candidate masks individually (one forward pass per candidate). With 128 points per side, expect ~200-800 candidate masks per image.
- **Data efficiency**: Works with 5-20 training examples

## Possible Future Improvements

- **Multiple prototypes**: Keep all training vectors instead of averaging, use max similarity (helps when training examples show multiple morphologies)
- **Full-image encoding**: Encode each image once and pool from the full feature map instead of cropping each candidate (faster but lower feature quality for small objects)
- **Negative examples**: Allow users to tag "not this" examples to reduce false positives
- **Size/shape priors**: Learn area distribution from training and filter candidates by size
- **Adaptive thresholding**: Use relative ranking (e.g., top 25%) instead of fixed threshold

## TODO / Future Work

- [ ] **Tiled image support**: Large microscopy images should be processed in tiles (like cellposesam's deeptile approach) rather than loading the entire image at once. This would reduce memory usage and allow processing of arbitrarily large images.
- [ ] **Multiple prototypes**: Keep all training feature vectors instead of averaging into a single prototype. Use max similarity or k-NN voting at inference. This would help when training examples show significant morphological variation.
- [ ] **Full-image encoding optimization**: Encode each inference image once and pool from the full feature map for each candidate mask, instead of cropping and re-encoding per candidate. Much faster but may reduce feature quality for small objects.
- [ ] **Negative examples**: Add a "Negative Tag" interface field so users can tag objects they do NOT want to match. Subtract negative similarity from positive similarity to reduce false positives.
- [ ] **Size/shape priors**: Learn area and aspect ratio distributions from training annotations and use them as an additional filter (e.g., reject candidates whose area is >2 std from training mean).
- [ ] **Adaptive thresholding**: Instead of a fixed similarity threshold, use relative ranking (e.g., keep top N% of candidates) or Otsu-style automatic thresholding on the similarity distribution.
- [ ] **Multi-scale feature extraction**: Extract features at multiple occupancy levels (e.g., 0.15, 0.25, 0.40) and concatenate for a richer feature vector. Helps when objects vary significantly in size.
- [ ] **Batch encoding**: Group multiple candidate crops into a batch tensor and encode them in a single forward pass through SAM2 for better GPU utilization.
- [ ] **Cache training prototype**: If the same training tag is used repeatedly, cache the prototype to avoid re-computing features on every run.
- [ ] **Similarity score as property**: Expose the similarity score as an annotation property so users can sort/filter results by confidence.
- [ ] **Support for point annotations as training**: Allow users to provide point prompts (not just polygon masks) as training examples, using SAM2's prompt-based segmentation to generate masks from points first.

## Files

| File | Purpose |
|------|---------|
| `entrypoint.py` | Worker logic: interface definition, feature extraction, inference pipeline |
| `Dockerfile` | x86_64 production build (CUDA 12.1, SAM2 checkpoints) |
| `Dockerfile_M1` | arm64/M1 Mac build (CUDA 11.8) |
| `environment.yml` | Conda environment specification |
| `tests/test_sam2_fewshot.py` | Unit tests for helper functions |
| `tests/Dockerfile_Test` | Test Docker image |
