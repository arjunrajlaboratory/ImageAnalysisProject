# Point Circle Intensity Mean

Computes the mean pixel intensity within a circular region around each point annotation.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Channel | channel | - | The image channel to measure intensity in |
| Radius | number | 3 | Radius of the outer sampling circle in pixels (min: 2, max: 10) |
| Radius2 | number | 1 | A second radius parameter (min: 0.5, max: 10); defined in the interface but not used in compute |

## Computed Properties

| Property | Description |
|----------|-------------|
| (single value) | Mean intensity of all pixels within the circle |

## How It Works

1. For each point annotation, the worker fetches the associated image.
2. A 0.5 pixel offset is subtracted from the point coordinates to convert from pixel-corner to pixel-center convention for `skimage.draw.disk`.
3. A disk of the specified `Radius` is drawn using `skimage.draw.disk`, and a boolean mask is created from the disk pixels.
4. The mean intensity of all pixels under the mask is computed and stored as a single float value.

## Notes

- The `Radius2` interface parameter is defined but **not used** in the `compute` function. Only `Radius` is read.
- Returns a single float value per annotation (not a named property dictionary).
- Annotations are processed one at a time, each fetching its own image via `get_image_for_annotation`.
- Does not filter annotations by tags.
- If the disk captures zero pixels, the annotation is silently skipped (guarded by `if rr and cc`).
- If the image for an annotation is `None`, that annotation is silently skipped.
