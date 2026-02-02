# CondensateNet Worker

This worker uses [CondensateNet](https://github.com/arjunrajlaboratory/condensatenet), a deep learning model for segmenting biomolecular condensates in brightfield microscopy images.

## How It Works

CondensateNet uses a Feature Pyramid Network (FPN) architecture with an EfficientNet encoder to detect and segment condensates. The pipeline:

1. **Preprocessing**: Normalizes the input image
2. **Inference**: Runs the FPN model to produce probability masks and flow fields
3. **Post-processing**: Converts predictions to instance segmentation using watershed-based methods

## Interface Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Batch XY/Z/Time** | Ranges for batch processing (e.g., "1-3, 5-8") | Current tile |
| **Probability Threshold** | Minimum confidence for detection (0-1) | 0.15 |
| **Min Size** | Minimum condensate size in pixels | 15 |
| **Max Size** | Maximum condensate size in pixels | 600 |
| **Smoothing** | Polygon simplification tolerance | 0.3 |
| **Padding** | Expand (+) or shrink (-) polygons in pixels | 0 |
| **Tile Size** | Size of tiles for processing large images | 1024 |
| **Tile Overlap** | Fraction of overlap between tiles (0-1) | 0.1 |

## Implementation Details

### Tiling Support

For large images, the worker uses [DeepTile](https://github.com/arjunrajlaboratory/DeepTile) to split the image into overlapping tiles, process each tile independently, and stitch the results back together. This allows processing of images larger than GPU memory.

Key tiling parameters:
- **Tile Size**: Controls memory usage. Smaller tiles use less memory but may miss large objects. Default 1024px works well for most cases.
- **Tile Overlap**: Fraction of overlap between adjacent tiles. Objects spanning tile boundaries are stitched together using the overlap region. A value of 0.1 (10%) means 102 pixels of overlap for 1024px tiles.

**Important**: Ensure your objects are smaller than the overlap region. If tile size is 1024 and overlap is 0.1, objects should be less than ~102 pixels in their longest dimension to guarantee correct stitching.

**Tile Size and Detection Consistency**: CondensateNet uses percentile-based normalization (0.5 to 99.5 percentile) on each tile independently. Smaller tiles have fewer pixels for computing these statistics, making normalization more sensitive to local intensity variations. This can cause detection differences between tile sizes—smaller tiles (e.g., 512) may produce more detections than larger tiles (e.g., 1024) due to more aggressive local normalization. For consistent results, use larger tile sizes (1024 recommended) when GPU memory permits.

### Image Padding for FPN Compatibility

The FPN architecture requires input dimensions to be divisible by 32 (due to 5 downsampling stages where 2^5 = 32). Images with non-conforming dimensions (e.g., 1022x1024) cause tensor size mismatches when combining feature maps from different pyramid levels.

The worker handles this by:
1. Padding each tile to the nearest dimensions divisible by 32 using reflection padding
2. Running inference on the padded tile
3. Cropping the output mask back to the original tile dimensions

This ensures the model works with arbitrary tile sizes without modifying the underlying CondensateNet package.

### Polygon Post-processing

After segmentation, polygons are processed in the following order:
1. **Stitching**: Objects spanning tile boundaries are merged using DeepTile's `stitch_polygons()`
2. **Padding**: If non-zero, polygons are dilated (positive) or eroded (negative) using Shapely's `buffer()`
3. **Smoothing**: Polygons are simplified using `simplify()` to reduce vertex count while preserving shape

### Output

The worker creates polygon annotations for each detected condensate, tagged with the user-specified tags.

## Model Location

The CondensateNet model weights are baked into the Docker image at `/models/condensatenet` for faster startup (no download required at runtime).
