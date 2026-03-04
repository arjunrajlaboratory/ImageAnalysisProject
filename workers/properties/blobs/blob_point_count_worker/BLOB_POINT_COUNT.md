# Blob Point Count

Counts the number of point annotations contained within each polygon annotation. Points can be filtered by tags and optionally counted across all Z slices.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Tags of points to count** | tags | (required) | Tags identifying which point annotations to count. |
| **Count points across all z-slices** | select | Yes | If "Yes", counts points from all Z slices within each polygon. If "No", only counts points on the polygon's own Z slice. |
| **Exact tag match?** | select | No | If "Yes", points must match tags exclusively. If "No", points matching any of the specified tags are counted. |

## Computed Properties

| Property | Description |
|----------|-------------|
| (scalar value) | Integer count of points inside the polygon. Stored directly as the property value (not nested in a dict). |

## How It Works

1. Retrieves all polygon annotations (filtered by the property's own tags) and all point annotations.
2. Points are filtered by the specified tags and tag match mode.
3. For each polygon, an R-tree spatial index is built from the relevant points (filtered by Time/XY, and optionally Z).
4. The R-tree is queried with the polygon's bounding box, then exact containment is checked with `polygon.contains(point)`.
5. The R-tree index is rebuilt only when the Time or XY location changes between consecutive polygons.

## Notes

- The R-tree spatial index provides efficient spatial queries, avoiding O(n*m) brute-force point-in-polygon checks.
- When "Count points across all z-slices" is "Yes", points are filtered by Time and XY only, allowing cross-Z counting. When "No", points must also match the polygon's Z location.
- The property value is a plain integer, not a dictionary of named properties.
