# Point to Nearest Connected Point Distance

Computes the Euclidean distance from each source point to the nearest connected child point that matches specified tags. Only considers points that are linked via connection annotations (parent-child relationships).

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Tags of points to measure distance to | tags | - | Tags identifying the target child point annotations (required) |
| Target tag match | select | Exact | Whether target annotations must match tags exactly ("Exact") or have any overlap ("Any") |

## Computed Properties

| Property | Description |
|----------|-------------|
| (single value) | 3D Euclidean distance to the nearest connected child point matching the target tags |

## How It Works

1. Source points are filtered by the top-level `params['tags']` filter.
2. For each source point, the worker fetches all connection annotations where the source is the **parent**.
3. The child annotations from those connections are retrieved individually from the server.
4. The child list is then filtered by the "Tags of points to measure distance to" parameter.
5. The 3D Euclidean distance (`sqrt(dx^2 + dy^2 + dz^2)`) is calculated from the source to each qualifying child, and the minimum distance is stored.

## Notes

- Unlike "Point to Nearest Point Distance", this worker only considers points that are **connected** to the source via connection annotations (parent-child relationships), not all points with matching tags.
- The source point must be the **parent** in the connection; child-to-parent distances are not computed.
- Distance is computed in 3D using the `z` field from `coordinates[0]`.
- A source point is never matched to itself (filtered by annotation `_id`).
- If no connected child point matches the tags (or distance remains infinity), no property value is stored for that source point.
- Each source point's connections are fetched individually from the server, which may be slow for large datasets.
- Location matching is not applied -- all connected children matching the tags are considered regardless of their Z, T, or XY position.
- Annotations are processed one at a time (no bulk upload).
