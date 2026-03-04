# Point to Nearest Point Distance

Computes the Euclidean distance from each source point annotation to the nearest target point annotation, with optional cross-Z and cross-T matching.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Tags of points to measure distance to | tags | - | Tags identifying the target point annotations (required) |
| Target tag match | select | Exact | Whether target annotations must match tags exactly ("Exact") or have any overlap ("Any") |
| Measure across Z | checkbox | false | If checked, target points are not required to share the same Z location |
| Measure across T | checkbox | false | If checked, target points are not required to share the same Time location |

## Computed Properties

| Property | Description |
|----------|-------------|
| (single value) | 3D Euclidean distance to the nearest target point (uses x, y, and z coordinates) |

## How It Works

1. Source points are filtered by the top-level `params['tags']` (the annotation tags set when running the property).
2. Target points are filtered by the worker interface "Tags of points to measure distance to" parameter.
3. For each source point, the target list is narrowed to those matching the required location attributes (XY always matches; Z and T matching is controlled by the checkboxes).
4. The 3D Euclidean distance (`sqrt(dx^2 + dy^2 + dz^2)`) is calculated from the source to every candidate target (excluding itself by `_id`), and the minimum distance is stored.

## Notes

- Distance is computed in 3D using the `z` field from `coordinates[0]`, not just 2D.
- XY location always must match; only Z and T matching can be relaxed via checkboxes.
- A source point is never matched to itself (filtered by annotation `_id`).
- If no target point is found (distance remains infinity), no property value is stored for that source point.
- Infinity values cannot be sent to the server via JSON, so annotations with no reachable target are simply omitted.
- Annotations are processed one at a time (no bulk upload).
