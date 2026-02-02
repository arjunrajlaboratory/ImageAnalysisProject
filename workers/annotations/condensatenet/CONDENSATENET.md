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

## Implementation Details

### Image Padding for FPN Compatibility

The FPN architecture requires input dimensions to be divisible by 32 (due to 5 downsampling stages where 2^5 = 32). Images with non-conforming dimensions (e.g., 1022x1024) cause tensor size mismatches when combining feature maps from different pyramid levels.

The worker handles this by:
1. Padding input images to the nearest dimensions divisible by 32 using reflection padding
2. Running inference on the padded image
3. Cropping the output mask back to the original dimensions

This ensures the model works with arbitrary image sizes without modifying the underlying CondensateNet package.

### Output

The worker creates polygon annotations for each detected condensate, tagged with the user-specified tags. Contours are extracted from the instance segmentation mask and simplified using the Shapely library.

## Model Location

The CondensateNet model weights are baked into the Docker image at `/models/condensatenet` for faster startup (no download required at runtime).
