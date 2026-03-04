# Connect Timelapse Worker

This worker connects annotations across time slices with support for bridging gaps in time. It links each object to its nearest neighbor in prior time points, preferring the most recent available match within a configurable gap window.

## How It Works

1. **Load annotations**: Fetches all point and polygon annotations from the dataset
2. **Filter by tag**: Selects annotations matching the user-specified tag
3. **Convert to centroids**: Extracts centroid coordinates for each annotation (point coordinates directly, polygon centroids via Shapely)
4. **Group spatially**: Groups objects by XY and Z position
5. **Process time slices**: Within each spatial group, iterates through time points in reverse order. For each time slice, gathers candidate parents from the preceding time points within the gap window
6. **Match with gap-aware logic**: Uses a cKDTree to find all candidate parents within the max distance, then selects the parent from the most recent available time point (breaking ties by distance)
7. **Create connections**: Uploads parent-child connection annotations tagged as "Time lapse connection"

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Object to connect tag** | tags | -- | Tag identifying which annotations to connect |
| **Connect across gaps** | number | 0 | Number of time points that can be bridged when an object has no match in the immediately preceding slice (0-10) |
| **Max distance** | number (pixels) | 20 | Maximum allowed distance between connected objects (0-1000). Pairs beyond this distance are not connected |

## Implementation Details

### Gap Bridging

The key feature distinguishing this worker from Connect Sequential is gap bridging. When "Connect across gaps" is set to N, the worker looks back up to N+1 time points for candidate parents. For example, with a gap of 2 and an object at T=5, the worker searches T=4, T=3, and T=2 for potential connections.

Among all candidates within the max distance, the worker prioritizes:
1. **Most recent time point first**: Candidates from later (closer) time points are preferred
2. **Closest distance second**: Among candidates at the same time point, the nearest one is selected

This ensures objects are connected to their most temporally proximate match, falling back to earlier time points only when no match is found in the immediately preceding slice.

### Spatial Grouping

Annotations are grouped by both XY position and Z slice. Connections are only made within the same XY and Z group -- cross-Z and cross-XY connections are never created.

### Matching Algorithm

The worker uses `cKDTree.query_ball_point` to find all parents within the max distance (rather than just the single nearest). From this candidate set, it selects the parent with the maximum time value (most recent), breaking ties by Euclidean distance. This is more sophisticated than a simple nearest-neighbor query because it accounts for temporal proximity.

### Output Tags

All created connections are tagged with "Time lapse connection" (hardcoded), regardless of the input object tag.

## Notes

- Line annotations are not currently supported as input; only point and polygon annotations are used.
- Polygons with fewer than 3 coordinates are skipped during processing.
- The default max distance is 20 pixels, which is much smaller than the other connection workers (1000 pixels). This reflects the expectation that timelapse tracking involves small frame-to-frame movements.
- If no objects with the specified tag are found, the worker sends a warning and returns without creating connections.
- If no tag is specified, the worker sends an error and raises a ValueError.
