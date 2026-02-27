# Rolling Ball Worker

Performs rolling ball background subtraction on selected channels using `skimage.restoration.rolling_ball` and uploads the corrected image to Girder.

## How It Works

1. Reads all frames from the source image
2. For each frame whose channel is in the selected set, estimates the background using the rolling ball algorithm and subtracts it from the original image
3. Writes all frames (corrected and unmodified) to a new TIFF and uploads it to the dataset

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Radius** | number | 20 | Radius of the rolling ball (range 0-100). Larger values estimate a smoother background. |
| **Channels to correct** | channelCheckboxes | -- | Select which channels to apply background subtraction to |

## Implementation Details

- The rolling ball algorithm estimates a local background by "rolling" a ball of the specified radius beneath the intensity surface. The estimated background is subtracted from the original, leaving foreground features intact.
- Only selected channels are processed; unselected channels pass through unchanged.
- Supports a preview mode (though the preview currently uses Gaussian blur, not rolling ball -- this appears to be a copy-paste artifact from the Gaussian blur worker).
- If the image has no `frames` metadata, falls back to processing a single frame.

## Notes

- The radius parameter controls sensitivity: small radii remove fine-grained background variations, while large radii only remove broad, slowly varying backgrounds.
- Useful for correcting uneven illumination in fluorescence microscopy images.
- Output metadata includes the tool name and radius used.
