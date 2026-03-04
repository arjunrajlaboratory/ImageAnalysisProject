# Point Metrics

Records the x and y coordinates of each point annotation as properties.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Point Metrics | notes | - | Descriptive text explaining that this worker documents point coordinates |

## Computed Properties

| Property | Description |
|----------|-------------|
| x | The x coordinate of the point annotation |
| y | The y coordinate of the point annotation |

## How It Works

For each point annotation, the worker reads the `x` and `y` fields from the annotation's first coordinate and stores them as float property values. No image data is accessed. Properties are uploaded in a single bulk call after all annotations are processed.

## Notes

- This is a purely geometric worker that does not require any image channel.
- Annotations are filtered by tags using the top-level `params['tags']` filter.
- The coordinates stored are the raw annotation coordinates (not pixel-center adjusted).
- Progress is reported at ~1% increments for large annotation sets.
- If no annotations match the filter, the worker exits silently.
