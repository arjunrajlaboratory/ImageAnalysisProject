# SAM Automatic Mask Generator

This worker uses Meta's [Segment Anything Model (SAM1)](https://github.com/facebookresearch/segment-anything) to automatically detect and segment all objects in an image without any user-provided prompts.

## How It Works

1. **Image Loading**: Loads two channels (the selected channel and channel 1) from the current tile position
2. **Auto-scaling**: Each channel is independently contrast-adjusted using 1st and 99.5th percentile normalization
3. **RGB Compositing**: Combines the channels into a pseudo-RGB image (phase in R/B, max of phase and fluor in G)
4. **SAM Inference**: Runs `SamAutomaticMaskGenerator` with 64 points per side to generate masks covering all detected objects
5. **Polygon Extraction**: Converts each binary mask to contours via `find_contours`, simplifies with Shapely, and uploads as polygon annotations

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Model** | select | `sam_vit_h_4b8939` | SAM1 ViT-H model checkpoint |
| **Use all channels** | checkbox | True | Whether to use all channels for compositing |
| **Padding** | number | 0 | Expand (+) or shrink (-) polygons in pixels (-20 to 20) |
| **Smoothing** | number | 0.3 | Polygon simplification tolerance (0 to 3) |

## Implementation Details

### Model

Uses the SAM1 ViT-H architecture (`vit_h`) with the `sam_vit_h_4b8939.pth` checkpoint. This is the largest SAM1 model variant.

### GPU Handling

Automatically uses CUDA if available, otherwise falls back to CPU. Note: there is a minor bug where `torch.cuda.is_available` (without parentheses) is used for the print statement, though the actual device selection correctly calls `torch.cuda.is_available()`.

### Mask Generation

The automatic mask generator uses 64 points per side as the grid density for generating candidate masks. This is hardcoded and not exposed as an interface parameter.

### Coordinate Handling

Contour coordinates from scikit-image (row, col) are swapped to annotation format (x, y) when creating annotation coordinates: `{"x": float(y), "y": float(x)}`.

## Notes

- This is a SAM1 worker. For the SAM2 equivalent with batch processing support, see `sam2_automatic_mask_generator`.
- The RGB compositing is tailored for phase contrast + fluorescence imaging and always loads channel 1 as the second channel regardless of the "Use all channels" setting.
- Does not support batch processing across XY/Z/Time positions (processes only the current tile).
