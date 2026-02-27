# H&E Deconvolution Worker

Performs color deconvolution on H&E-stained (Hematoxylin and Eosin) RGB images using the HED color space transform from scikit-image. Separates the stain components and uploads the deconvolved result to Girder.

## How It Works

1. Groups frames by spatial position (XY, Z, Time), collecting all 3 RGB channels per position
2. Assembles each position's channels into an RGB image
3. Applies `skimage.color.rgb2hed` to transform into the HED (Hematoxylin-Eosin-DAB) color space
4. Rescales each deconvolved channel's intensity using `skimage.exposure.rescale_intensity` with a percentile-based input range
5. Writes the deconvolved channels as a new TIFF and uploads it to the dataset

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Max percentile** | number | 99 | Upper percentile for intensity rescaling (range 0-100). Controls clipping of bright outliers. |

## Implementation Details

- Requires exactly 3 channels (RGB). If the image does not have 3 channels, the worker sends an error and exits.
- The HED transform produces continuous-valued output. Each channel is rescaled to 0-255 uint8 using `rescale_intensity` with `in_range=(0, percentile_value)` and `out_range=(0, 255)`.
- The max percentile parameter controls how aggressively bright pixels are clipped during rescaling. Lower values increase contrast but clip more of the bright end.
- Frames are grouped by unique (XY, Z, Time) positions to correctly assemble multi-channel data before the RGB-to-HED conversion.
- If the image has no `frames` metadata (single frame), the worker sends an error and exits.

## Notes

- The output image retains the original channel names but contains HED-space values, not the original RGB values. Channel 0 corresponds to Hematoxylin, channel 1 to Eosin, and channel 2 to DAB.
- This is specifically designed for histology images with H&E staining. It will not produce meaningful results on fluorescence or other image types.
- Output metadata includes the tool name.
