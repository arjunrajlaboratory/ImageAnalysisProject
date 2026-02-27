# DeepCell Worker

This worker uses [DeepCell's Mesmer](https://github.com/vanvalenlab/deepcell-tf) model for whole-cell and nuclear segmentation in multiplexed tissue imaging.

## How It Works

1. **Image Loading**: Fetches a single-channel image from the dataset for the selected tile position
2. **Tiling**: Uses [DeepTile](https://github.com/arjunrajlaboratory/DeepTile) to split the image into 640x640 tiles with padding
3. **Input Preparation**: Stacks the image with itself to create a 2-channel input (Mesmer expects a nuclear + membrane pair)
4. **Inference**: Runs the Mesmer segmentation model on each tile
5. **Stitching**: Merges tile-level masks into a single label image using `stitch_masks()`
6. **Polygon Extraction**: Converts the label image to vector polygons using `rasterio.features.shapes()`
7. **Upload**: Creates polygon annotations one at a time, with a hard cap of 1000 annotations

## Interface Parameters

This worker does not define a custom `interface()` function. It uses only the standard parameters provided by the NimbusImage platform:

| Parameter | Type | Description |
|-----------|------|-------------|
| **channel** | (platform) | Image channel to segment |
| **tags** | (platform) | Tags to apply to output annotations |
| **tile** | (platform) | Tile position (XY, Z, Time) to process |
| **assignment** | (platform) | Assignment coordinates for output annotation location |

## Implementation Details

- **GPU Support**: Built on NVIDIA CUDA 11.8 base image. DeepCell/Mesmer uses TensorFlow for GPU-accelerated inference.
- **Model Pre-download**: The Mesmer model is downloaded during Docker build by instantiating `Mesmer()` in `download_models.py`, so no network access is needed at runtime.
- **Tiling**: Uses DeepTile with a fixed 640x640 tile size. Tiles are padded and masks are stitched back together after inference.
- **Duplicate Channel Input**: Mesmer expects a 2-channel input (nuclear + membrane). This worker duplicates the single input channel for both, which means it effectively runs nuclear segmentation only.
- **Annotation Limit**: A hard cap of 1000 annotations is enforced to avoid flooding the server. This is marked as a TODO in the code.
- **Single Annotation Upload**: Annotations are uploaded one at a time via `createAnnotation()` rather than in batch via `createMultipleAnnotations()`.

## Notes

- Mesmer is designed for multiplexed tissue imaging and works best with nuclear markers. For optimal whole-cell segmentation, a separate membrane channel would be needed, but this worker does not currently support that.
- The 1000-annotation limit may silently truncate results on dense images. Check the worker logs for the expected annotation count.
- This worker does not support batch processing over XY/Z/Time dimensions. It processes only the current tile position.
- No user-configurable parameters are exposed beyond the standard platform controls (channel, tags, tile).
