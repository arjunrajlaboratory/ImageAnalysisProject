# Histogram Matching Worker

Normalizes intensity distributions across frames by matching the histogram of each selected channel to a user-specified reference frame, using `skimage.exposure.match_histograms`. Uploads the normalized image to Girder.

## How It Works

1. Loads the reference image for each selected channel at the specified (XY, Z, Time) coordinate
2. Iterates through all frames; for frames in selected channels, transforms the intensity histogram to match the corresponding reference image
3. Unselected channels pass through unchanged
4. Writes the normalized result to a new TIFF and uploads it to the dataset

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Reference XY Coordinate** | text | 1 (first position) | XY coordinate of the reference image (1-indexed). Leave empty for first position. |
| **Reference Z Coordinate** | text | 1 (first position) | Z coordinate of the reference image (1-indexed). Leave empty for first position. |
| **Reference Time Coordinate** | text | 1 (first position) | Time coordinate of the reference image (1-indexed). Leave empty for first position. |
| **Channels to correct** | channelCheckboxes | -- | Select which channels to normalize |

## Implementation Details

- Reference coordinates are 1-indexed in the UI and converted to 0-indexed internally (empty string defaults to index 0).
- Each selected channel gets its own reference image, loaded once before processing begins.
- If no channels are selected, the worker sends an error and exits.
- If the image has only a single frame (no `frames` metadata), the worker sends an error and exits, since histogram matching requires multiple frames.

## Notes

- Useful for correcting intensity drift across time-lapse acquisitions or normalizing illumination differences across XY positions.
- The reference frame itself will be matched to itself (a no-op), preserving its original intensities.
- Output metadata includes the tool name and reference coordinates used.
