# Line Length

Computes the total Euclidean length of line annotations, including 3D distance using x, y, and z coordinates.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Line Length | notes | N/A | Descriptive note: "Computes the length of lines." |

## Computed Properties

| Property | Description |
|----------|-------------|
| Length | Total length of the line, computed as the sum of Euclidean distances between consecutive coordinate points in 3D (x, y, z). |

## How It Works

1. Fetches all line annotations from the dataset, filtered by the selected tags.
2. For each line annotation, iterates through consecutive pairs of coordinates.
3. Computes the 3D Euclidean distance between each pair: `sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)`.
4. Sums the segment distances to get the total line length.
5. Stores the total length as a single scalar property value on each annotation.

## Notes

- The length calculation uses all three spatial dimensions (x, y, z) from the annotation coordinates, not just the 2D projection.
- If no line annotations match the selected tags, the worker returns immediately without producing output.
- Each annotation's property value is uploaded individually via `add_annotation_property_values`.
