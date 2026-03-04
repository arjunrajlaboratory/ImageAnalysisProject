# Blob Intensity

Computes pixel intensity statistics for polygon annotations in a specified channel. Supports computing across multiple Z planes with nested output.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Channel** | channel | (required) | Channel to compute pixel intensities in. Does not need to match the annotation layer. |
| **Z planes** | text | (empty) | Z positions to compute intensities for (e.g., "1-3, 5-8"). Leave empty to use the annotation's own Z plane. |
| **Additional percentiles** | text | (empty) | Comma-separated percentile values to compute (e.g., "10, 45, 90"). Leave empty for default 25th/75th only. |

## Computed Properties

| Property | Description |
|----------|-------------|
| MeanIntensity | Mean pixel intensity within the polygon |
| MaxIntensity | Maximum pixel intensity |
| MinIntensity | Minimum pixel intensity |
| MedianIntensity | Median pixel intensity |
| 25thPercentileIntensity | 25th percentile intensity |
| 75thPercentileIntensity | 75th percentile intensity |
| TotalIntensity | Sum of all pixel intensities |
| {N}thPercentileIntensity | Additional percentile intensities (if specified) |

## How It Works

1. Annotations are grouped by their (Time, Z, XY) location to minimize image loading.
2. For each annotation, the polygon is rasterized using `skimage.draw.polygon` with the standard 0.5-pixel offset applied to coordinates.
3. Pixel intensities are extracted from the rasterized mask region.
4. If **Z planes** are specified, properties are stored as nested dictionaries keyed by Z plane (e.g., `{"z001": value, "z002": value}`). Otherwise, properties are flat scalar values.

## Notes

- Polygons with fewer than 3 vertices or empty masks are skipped with a warning.
- Z plane input uses 1-based indexing (user-facing); internally converted to 0-based. Out-of-range planes are excluded with a warning.
- Additional percentile values must be between 0 and 100 (exclusive).
