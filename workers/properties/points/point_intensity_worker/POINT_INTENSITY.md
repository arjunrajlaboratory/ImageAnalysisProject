# Point Intensity

Computes the pixel intensity at each point annotation's exact location in the specified channel.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Channel | channel | - | The image channel to sample intensity from |

## Computed Properties

| Property | Description |
|----------|-------------|
| (single value) | Integer pixel intensity at the point's location |

## How It Works

For each point annotation, the worker retrieves the image for that annotation, rounds the point's coordinates to the nearest pixel (`round(y)`, `round(x)`), and reads the single pixel intensity value at that location. The result is stored as an integer.

## Notes

- Coordinates are rounded to the nearest pixel using `round()` (no 0.5 pixel offset applied).
- Returns a single integer value per annotation (not a named property dictionary).
- Annotations are processed one at a time, each fetching its own image via `get_image_for_annotation`.
- Does not filter annotations by tags.
- If the image for an annotation is `None`, that annotation is silently skipped.
