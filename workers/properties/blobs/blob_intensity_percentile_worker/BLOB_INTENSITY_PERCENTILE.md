# Blob Intensity Percentile

Computes a single user-specified percentile of pixel intensity for polygon annotations in a specified channel. A lightweight alternative to Blob Intensity when only one percentile value is needed.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Channel** | channel | (required) | Channel to compute pixel intensities in. Does not need to match the annotation layer. |
| **Percentile** | number | 50 | Percentile value to compute (0 to 99.99999). |

## Computed Properties

| Property | Description |
|----------|-------------|
| {N}thPercentileIntensity | Intensity at the specified percentile. Property name includes the percentile value (e.g., "50.0thPercentileIntensity"). |

## How It Works

1. Annotations are grouped by their (Time, Z, XY) location to minimize image loading.
2. For each annotation, the polygon is rasterized using `skimage.draw.polygon` with the standard 0.5-pixel offset.
3. `numpy.percentile` is computed on the extracted pixel intensities at the user-specified percentile.

## Notes

- Polygons with fewer than 3 vertices or empty masks are silently skipped (no warning sent).
- Unlike Blob Intensity, this worker does not support multi-Z-plane mode; it always uses the annotation's own Z plane.
- The property name dynamically includes the percentile value, so different runs with different percentiles produce differently named properties.
