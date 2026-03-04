# Blob Colony Two Color Intensity

Computes intensity statistics for polygon annotations across two channels simultaneously, using a combined threshold mask to focus on bright pixels across both channels. Designed for colony-level analysis where signal from either channel indicates relevant regions.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Channel 1** | channel | (required) | First channel for intensity measurement. |
| **Channel 2** | channel | (required) | Second channel for intensity measurement. |
| **Threshold percentile** | number | 50 | Percentile threshold (0-100, step 0.1). Pixels above this percentile in either channel are included in the combined mask for MeanColonyIntensity. |

## Computed Properties

Properties are nested under `Channel 1` and `Channel 2` keys:

| Property | Description |
|----------|-------------|
| MeanColonyIntensity | Mean intensity of pixels above the threshold percentile in **either** channel (combined OR mask). |
| MedianIntensity | Median intensity of all pixels in the polygon. |
| 25thPercentileIntensity | 25th percentile intensity of all pixels in the polygon. |
| 40thPercentileIntensity | 40th percentile intensity of all pixels in the polygon. |
| 75thPercentileIntensity | 75th percentile intensity of all pixels in the polygon. |

## How It Works

1. Annotations are grouped by (Time, Z, XY) location. Both channel images are loaded per location.
2. For each annotation, a binary mask is created using `skimage.draw.polygon2mask`.
3. The threshold percentile is computed independently for each channel's masked pixels.
4. A combined mask is created: pixels above the threshold in **either** channel (logical OR).
5. MeanColonyIntensity is computed using only the combined-mask pixels. Other statistics use all polygon pixels.

## Notes

- This worker uses `polygon2mask` (not `draw.polygon`), and constructs the polygon array differently: `list(coordinate.values())[1::-1]` reverses the first two coordinate values.
- The combined mask approach captures pixels that are bright in at least one channel, making it suitable for analyzing overlapping but not identical expression patterns in colonies.
- Percentile statistics (median, 25th, 40th, 75th) are computed on the full polygon region, not the thresholded subset.
