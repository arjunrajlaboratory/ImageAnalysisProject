# Gaussian Blur Worker

Applies a Gaussian blur filter to selected channels of an image and uploads the processed result as a new image to Girder.

## How It Works

1. Reads all frames from the source image
2. For each frame whose channel is in the selected set, applies `skimage.filters.gaussian` with the specified sigma
3. Multiplies the result by the dtype max value to preserve the original intensity range
4. Writes all frames (blurred and unmodified) to a new TIFF and uploads it to the dataset

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Sigma** | number | 20 | Sigma value for the Gaussian blur filter (range 0-100) |
| **Channel** | channel | 0 | The channel to blur (used for preview only) |
| **All channels** | channelCheckboxes | -- | Select which channels to blur in the output image |

## Implementation Details

- The worker preserves the original image dtype. For integer types, the Gaussian output (which is normalized to 0-1) is scaled by the dtype's max value before casting back.
- Only channels selected via the "All channels" checkboxes are blurred; all other channels are passed through unchanged.
- Supports a live preview mode that shows the blurred result for the current tile as an RGBA overlay.
- If the image has no `frames` metadata, falls back to processing a single frame.

## Notes

- The "Channel" parameter is only used for the preview function, not for the main compute. The "All channels" checkboxes control which channels are actually processed.
- Output metadata includes the tool name, sigma, and channel used.
