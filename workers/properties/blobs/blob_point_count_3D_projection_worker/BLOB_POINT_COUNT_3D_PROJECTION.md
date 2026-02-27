# Blob Point Count 3D Projection

Counts the number of point annotations contained within each polygon annotation, always projecting across all Z slices. A simpler variant of Blob Point Count without the Z-slice toggle.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Tags of points to count** | tags | (required) | Tags identifying which point annotations to count. |
| **Exact tag match?** | select | Yes | If "Yes", points must match tags exclusively. If "No", points matching any of the specified tags are counted. |

## Computed Properties

| Property | Description |
|----------|-------------|
| (scalar value) | Integer count of points inside the polygon. Stored directly as the property value (not nested in a dict). |

## How It Works

1. Retrieves all polygon annotations and all point annotations.
2. Points are filtered by the specified tags and tag match mode.
3. Points are filtered by Time and XY only (always projects across all Z slices).
4. For each polygon, an R-tree spatial index is used for efficient containment testing.
5. The R-tree is rebuilt only when the Time or XY location changes between consecutive polygons.

## Notes

- This worker always counts across all Z slices, unlike the standard Blob Point Count worker which offers a toggle.
- The default for "Exact tag match?" is "Yes" (compared to "No" in the standard Blob Point Count worker).
- The property value is a plain integer, not a dictionary of named properties.
