# Point Threshold Intensity Mean

Computes the mean intensity of the thresholded region containing each point annotation, using Otsu's method on a local crop around the point.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Channel | channel | - | The image channel to measure intensity in |

## Computed Properties

| Property | Description |
|----------|-------------|
| (single value) | Mean intensity of pixels in the thresholded object at the point's location |

## How It Works

1. For each point annotation, the worker fetches the associated image.
2. A 25x25 pixel crop is extracted centered on the point (clipped to image boundaries).
3. Otsu's threshold (`skimage.filters.threshold_otsu`) is applied to the crop to create a binary image.
4. Connected components are labeled using `skimage.measure.label`.
5. The label at the point's center position is selected, isolating just the connected object the point falls within.
6. The mean intensity of the pixels belonging to that labeled object is computed and stored as a single float value.

## Notes

- The block size for cropping is hardcoded at 25 pixels and cannot be changed via the interface.
- Coordinates are rounded using `round()` (no 0.5 pixel offset).
- If the point falls on a background pixel (below the Otsu threshold), the labeled region at that position will be the background component, and the mean intensity of the background region in the crop will be returned.
- Does not filter annotations by tags.
- Returns a single float value per annotation (not a named property dictionary).
- Annotations are processed one at a time, each fetching its own image via `get_image_for_annotation`.
- If the image for an annotation is `None`, that annotation is silently skipped.
