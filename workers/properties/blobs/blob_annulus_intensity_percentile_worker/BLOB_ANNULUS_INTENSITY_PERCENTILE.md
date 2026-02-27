# Blob Annulus Intensity Percentile

Computes a single user-specified percentile of pixel intensity in an annular region surrounding each polygon annotation. A lightweight alternative to Blob Annulus Intensity when only one percentile is needed.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Channel** | channel | (required) | Channel to compute pixel intensities in. Does not need to match the annotation layer. |
| **Radius** | number | 10 | Width of the annulus in pixels (0-200). Defines how far to dilate outward from the polygon boundary. |
| **Percentile** | number | 50 | Percentile value to compute (0 to 99.99999). |

## Computed Properties

| Property | Description |
|----------|-------------|
| {N}thPercentileIntensity | Intensity at the specified percentile in the annulus region. Property name includes the percentile value (e.g., "50.0thPercentileIntensity"). |

## How It Works

1. Annotations are grouped by their (Time, Z, XY) location to minimize image loading.
2. For each annotation, the polygon is rasterized and then dilated by the specified radius using Shapely's `buffer()`.
3. The annulus is the set difference between the dilated and original polygon pixel coordinates.
4. `numpy.percentile` is computed on the annulus pixel intensities at the user-specified percentile.

## Notes

- Unlike Blob Annulus Intensity, this worker does not support multi-Z-plane mode; it always uses the annotation's own Z plane.
- Polygons with fewer than 3 vertices or empty annulus/masks are silently skipped.
- The image is explicitly squeezed (`.squeeze()`) to handle extra dimensions from `getRegion`.
- The property name dynamically includes the percentile value.
