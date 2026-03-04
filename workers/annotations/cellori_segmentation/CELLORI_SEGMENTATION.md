# Cellori Segmentation Worker

This worker uses [Cellori](https://github.com/zjniu/Cellori) for cell segmentation, supporting both nuclear and cytoplasmic channels with batch processing across XY, Z, and Time dimensions.

## How It Works

1. **Image Loading**: Fetches nuclear and/or cytoplasmic channel images for each tile position in the batch
2. **Input Preparation**: If both channels are provided, stacks them as a 2-channel input. If only cytoplasm is provided, uses it as a single-channel input. Nuclear-only mode is not supported.
3. **Inference**: Runs the Cellori `cyto` model with the specified diameter parameter
4. **Polygon Extraction**: Converts the predicted mask to vector polygons using `rasterio.features.shapes()`
5. **Upload**: Creates polygon annotations for each detected cell

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Nuclei Channel** | channel | -1 | Channel containing nuclear stain. Set to -1 to disable. |
| **Cytoplasm Channel** | channel | -1 | Channel containing cytoplasm/membrane stain. Set to -1 to disable. |
| **Diameter** | number | 10 | Expected cell diameter in pixels (0-200) |
| **Batch XY** | text | (current tile) | XY positions to process, e.g. "1-3, 5-8" |
| **Batch Z** | text | (current tile) | Z slices to process, e.g. "1-3, 5-8" |
| **Batch Time** | text | (current tile) | Time points to process, e.g. "1-3, 5-8" |

## Implementation Details

- **GPU Support**: Built on NVIDIA CUDA 11.8 base image. Cellori uses JAX with CUDA 11 for GPU-accelerated inference.
- **Channel Stacking**: When both nuclei and cytoplasm channels are provided, the cytoplasm image is placed first in the stack (`np.stack((cytoplasm, nuclei))`), matching Cellori's expected input order.
- **Model**: Always uses the `cyto` model with `batch_size=1`.
- **Batch Processing**: Supports processing multiple XY/Z/Time positions via text-based range inputs (e.g., "1-3, 5-8"). Uses 1-based indexing in the interface, converted internally to 0-based. If no batch ranges are specified, processes only the current tile position.
- **Single Annotation Upload**: Annotations are uploaded one at a time via `createAnnotation()` rather than in batch.

## Notes

- At least one of Nuclei Channel or Cytoplasm Channel must be set to a valid channel (not -1). If only Nuclei Channel is set (without Cytoplasm Channel), the `image` variable will be `None` and the worker will fail, since the code only assigns to `image` when cytoplasm is present.
- The `diameter` parameter significantly affects segmentation quality. It should approximate the typical cell diameter in pixels in your images.
- Unlike some other workers, Cellori does not use tiling -- the entire image is processed in a single pass.
