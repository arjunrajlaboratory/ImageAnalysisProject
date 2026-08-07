# StarDist Worker

This worker uses [StarDist](https://github.com/stardist/stardist) to segment nuclei and cells using star-convex polygon detection. It supports pretrained models for both fluorescence and H&E-stained images.

## How It Works

1. **Image Loading**: Fetches a single-channel image from the dataset for the selected tile position
2. **Normalization**: Converts the image to float32 and normalizes to 0-1 range
3. **Inference**: Runs StarDist2D to predict instance labels using star-convex polygon detection with configurable probability and NMS thresholds
4. **Polygon Extraction**: Converts the label image to vector polygons using `rasterio.features.shapes()`
5. **Post-processing**: Optionally applies padding (dilation/erosion via Shapely `buffer()`) and smoothing (vertex reduction via `simplify()`)
6. **Upload**: Creates polygon annotations for each detected object

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Model** | select | `2D_versatile_fluo` | Pretrained model: `2D_versatile_fluo` (fluorescence) or `2D_versatile_he` (H&E histology) |
| **Channel** | channel | 0 | Image channel to segment (required) |
| **Probability Threshold** | number | 0.5 | Minimum detection confidence (0-1). Lower values detect more objects but increase false positives |
| **NMS Threshold** | number | 0.4 | Non-maximum suppression overlap threshold (0-1). Lower values remove more overlapping detections |
| **Padding** | number | 0 | Expand (+) or shrink (-) polygons in pixels (-20 to 20) |
| **Smoothing** | number | 1 | Polygon simplification tolerance (0-10). Higher values produce simpler polygons with fewer vertices |

## Implementation Details

- **GPU Support**: Built on NVIDIA CUDA 11.8 base image. StarDist uses TensorFlow for GPU-accelerated inference. The runtime stage must keep a `cudnn8` tag: TensorFlow 2.11 predates the `tensorflow[and-cuda]` extra (2.14), so its wheel bundles no `nvidia-*` packages and dlopens `libcudnn.so.8` from the image itself. Dropping to a plain `-runtime` tag would not fail the build — TF would log "Could not load dynamic library libcudnn.so.8" and silently fall back to CPU. (The torch-based ML workers are the opposite: their wheels ship their own cuDNN, so they use the plain `-runtime` tag.)
- **Model Pre-download**: Both pretrained models (`2D_versatile_fluo` and `2D_versatile_he`) are downloaded and cached during Docker build via `download_models.py`, so no network access is needed at runtime.
- **Polygon Conversion**: Uses `rasterio.features.shapes()` with a transform that maps pixel coordinates directly to image space. Falls back to default transform if the explicit one fails.
- **No Tiling**: The worker processes the entire image in a single pass (no tile-based processing). Very large images may require significant GPU memory.

### Model selection validation

`compute()` validates the saved `Model` selection with
`annotation_tools.get_required_select()` before loading the model. A saved tool
config can hold `null` for a `select` field even though the interface defines a
default, and a config saved against an older worker image can name a model that
is no longer offered. Previously a `null` model was passed straight to
`StarDist2D.from_pretrained()`. Invalid selections now fail fast with
`sendError`; the fix is to re-select the model in the tool settings and save the
tool again. The selection is validated against the static pretrained-model list
(`MODELS`).

## Notes

- The `2D_versatile_fluo` model works well for fluorescence microscopy nuclei (e.g., DAPI). The `2D_versatile_he` model is designed for H&E-stained histology images.
- StarDist's star-convex polygon approach is particularly well-suited for convex or near-convex objects like nuclei. It may not perform well on highly irregular or concave cell shapes.
- This worker does not support batch processing over XY/Z/Time dimensions. It processes only the current tile position.
