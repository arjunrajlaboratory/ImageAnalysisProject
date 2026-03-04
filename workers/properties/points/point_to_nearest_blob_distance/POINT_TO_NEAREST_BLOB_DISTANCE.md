# Point to Nearest Blob Distance

Computes the distance from each point annotation to the nearest blob (polygon) annotation, with an option to measure to the blob's centroid or edge. Can optionally create connection annotations between matched pairs.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Blob tags | tags | - | Tags identifying the target blob annotations (required) |
| Distance type | select | Centroid | Whether to measure distance to the blob's centroid ("Centroid") or nearest edge ("Edge") |
| Create connection | checkbox | false | If checked, creates a parent-child connection between each point and its nearest blob |

## Computed Properties

| Property | Description |
|----------|-------------|
| (single value) | Distance from the point to the nearest blob (centroid or edge, depending on selection) |

## How It Works

1. Source points are filtered by the top-level `params['tags']` filter.
2. Target blobs (polygons) are filtered by the "Blob tags" interface parameter using non-exclusive matching.
3. For each source point, only blobs sharing the same `location` (XY, Z, Time) are considered.
4. Distance is calculated using Shapely geometry:
   - **Centroid**: distance from the point to the blob polygon's centroid.
   - **Edge**: distance from the point to the blob polygon's boundary.
5. The minimum distance and nearest blob are recorded.
6. If "Create connection" is enabled, a parent-child connection is created with the blob as parent and the point as child, tagged with the union of both annotations' tags.

## Notes

- Distance is computed in 2D (x, y only) using Shapely geometry operations.
- Blob coordinates are used directly as Shapely Polygon vertices (no coordinate swap).
- Point coordinates are used directly as a Shapely Point (no coordinate swap).
- Location must match exactly (XY, Z, and Time) for a point-blob pair to be considered. There is no option to match across Z or T.
- If no blob is found at the same location, no property value or connection is created for that point.
- Connections are uploaded in a single bulk call after all distances are computed.
- Progress is reported per point annotation.
