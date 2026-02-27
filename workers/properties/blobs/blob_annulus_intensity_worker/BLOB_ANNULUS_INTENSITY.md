# Blob Annulus Intensity

Computes pixel intensity statistics in an annular region surrounding each polygon annotation. The annulus is formed by dilating the polygon by a specified radius and subtracting the original polygon area. Supports multi-Z-plane mode.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Channel** | channel | (required) | Channel to compute pixel intensities in. Does not need to match the annotation layer. |
| **Radius** | number | 10 | Width of the annulus in pixels (0-200). Defines how far to dilate outward from the polygon boundary. |
| **Z planes** | text | (empty) | Z positions to compute intensities for (e.g., "1-3, 5-8"). Leave empty to use the annotation's own Z plane. |
| **Additional percentiles** | text | (empty) | Comma-separated percentile values to compute (e.g., "10, 45, 90"). Leave empty for default 25th/75th only. |

## Computed Properties

| Property | Description |
|----------|-------------|
| MeanIntensity | Mean pixel intensity in the annulus |
| MaxIntensity | Maximum pixel intensity in the annulus |
| MinIntensity | Minimum pixel intensity in the annulus |
| MedianIntensity | Median pixel intensity in the annulus |
| 25thPercentileIntensity | 25th percentile intensity in the annulus |
| 75thPercentileIntensity | 75th percentile intensity in the annulus |
| TotalIntensity | Sum of all pixel intensities in the annulus |
| {N}thPercentileIntensity | Additional percentile intensities (if specified) |

## How It Works

1. Annotations are grouped by location to minimize image loading.
2. For each annotation, the polygon is rasterized to get the original pixel coordinates.
3. The polygon is dilated by the specified radius using Shapely's `buffer()`, then rasterized.
4. The annulus is computed as the set difference: dilated coordinates minus original coordinates.
5. Intensity statistics are computed on the annulus pixels only.
6. If Z planes are specified, properties are stored as nested dictionaries keyed by Z plane (e.g., `{"z001": value}`).

## Notes

- The annulus is clipped to the image boundaries; pixels outside the image are excluded.
- If the annulus has zero pixels (e.g., polygon fills the image), the annotation is skipped with a warning.
- The 0.5-pixel offset is applied to annotation coordinates for rasterization consistency.
- Z plane input uses 1-based indexing (user-facing); internally converted to 0-based.
