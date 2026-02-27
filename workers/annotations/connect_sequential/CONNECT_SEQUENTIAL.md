# Connect Sequential Worker

This worker connects annotations sequentially across time points or Z slices by linking each object to its nearest neighbor in the immediately preceding slice. It is useful for tracking objects that move or change across time or Z.

## How It Works

1. **Load annotations**: Fetches all point and polygon annotations from the dataset
2. **Filter by tag**: Selects annotations matching the user-specified tag
3. **Convert to centroids**: Extracts centroid coordinates for each annotation (point coordinates directly, polygon centroids via Shapely)
4. **Sort and iterate**: Sorts objects in descending order along the connection axis (Time or Z), then for each object finds the nearest neighbor in the previous slice
5. **Create connections**: Each object becomes a "child" connected to its nearest "parent" in the preceding slice, with the parent being the earlier time point or lower Z

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Object to connect tag** | tags | -- | Tag identifying which annotations to connect |
| **Connect sequentially across** | select | Time | Whether to connect across "Time" or "Z" slices |
| **Max distance (pixels)** | number | 1000 | Maximum allowed distance between connected objects (0-5000). Pairs beyond this distance are not connected |

## Implementation Details

### Grouping Behavior

Annotations are always grouped by XY position. When connecting across Time, annotations are additionally grouped by Z (so only objects in the same Z slice are connected). When connecting across Z, annotations are additionally grouped by Time.

### Sequential Matching

Objects are sorted in descending order along the chosen axis. For each object, the worker looks only at objects in the immediately preceding slice (Time - 1 or Z - 1) and finds the nearest one using a cKDTree built from centroids. This means each object connects to at most one parent in the previous slice.

### Centroid-Only Distance

Unlike the Connect to Nearest worker, this worker always uses centroid-based distance. For polygon annotations, the centroid is computed via Shapely. Point annotations use their coordinates directly.

### Performance Note

The current implementation builds a new cKDTree for each individual object rather than batching all objects in a time/Z slice together. This is less efficient for large datasets but produces correct results. A TODO in the code notes this could be optimized.

### Output Tags

Created connections are tagged with the same tag(s) as the input objects.

## Notes

- Line annotations are not currently supported as input; only point and polygon annotations are used.
- Polygons with fewer than 3 coordinates are skipped during processing.
- Unlike Connect to Nearest, this worker does not support edge-based distance, connection restrictions (touching/within), or gap bridging across multiple slices. For gap bridging across time, use the Connect Timelapse worker instead.
