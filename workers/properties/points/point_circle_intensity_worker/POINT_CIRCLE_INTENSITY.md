# Point Circle Intensity

Computes summary intensity statistics for all pixels within a circular region around each point annotation.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Point Intensity | notes | - | Descriptive text explaining the worker's behavior |
| Channel | channel | - | The image channel to measure intensities in (required) |
| Radius | number | 1 | Radius of the sampling circle in pixels (min: 0.5, max: 10) |

## Computed Properties

| Property | Description |
|----------|-------------|
| MeanIntensity | Mean of all pixel intensities in the circle |
| MaxIntensity | Maximum pixel intensity in the circle |
| MinIntensity | Minimum pixel intensity in the circle |
| MedianIntensity | Median pixel intensity in the circle |
| 25thPercentileIntensity | 25th percentile of pixel intensities in the circle |
| 75thPercentileIntensity | 75th percentile of pixel intensities in the circle |
| TotalIntensity | Sum of all pixel intensities in the circle |

## How It Works

1. Annotations are grouped by their location (Time, Z, XY) so that annotations sharing the same location reuse a single image fetch.
2. For each annotation, a disk of the specified radius is drawn centered on the point using `skimage.draw.disk`.
3. A 0.5 pixel offset is subtracted from coordinates to convert from pixel-corner to pixel-center convention for scikit-image.
4. All seven intensity statistics are computed from the pixels within the disk and uploaded in bulk.

## Notes

- Annotations are filtered by tags using the top-level `params['tags']` filter.
- If the radius is 0.5 (the minimum), the disk typically captures just the single pixel at the point's location.
- If the disk falls partially outside the image bounds, `draw.disk` clips to the image shape, so only in-bounds pixels are included.
- If the disk captures zero pixels (e.g., point is outside image), the annotation is silently skipped.
- Progress is reported at ~1% increments for large annotation sets.
- Properties are sent to the server in a single bulk call after all annotations are processed.
