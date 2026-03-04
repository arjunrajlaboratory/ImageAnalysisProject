# Connect to Nearest Worker

This worker connects annotations to their nearest neighbors by creating connection (line) annotations from child objects to their closest parent objects. It supports both point and polygon annotations, with options for centroid-based or edge-based distance matching.

## How It Works

1. **Load annotations**: Fetches all point and polygon annotations from the dataset
2. **Filter by tags**: Separates annotations into parent and child sets based on user-specified tags
3. **Group by location**: Groups annotations by XY position, and optionally by Z and Time (unless cross-Z/T connection is enabled)
4. **Find nearest parents**: Within each group, uses a cKDTree (centroid mode) or GeoPandas `sjoin_nearest` (edge mode) to find the closest parent for each child
5. **Create connections**: Uploads parent-child connection annotations to the server

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Parent tag** | tags | -- | Tag identifying parent annotations to connect to |
| **Child tag** | tags | -- | Tag identifying child annotations to connect from |
| **Connect across Z** | select | No | When "Yes", allows connections between objects on different Z slices |
| **Connect across T** | select | No | When "Yes", allows connections between objects at different time points |
| **Connect to closest centroid or edge** | select | Centroid | Whether distance is measured between centroids or nearest polygon edges |
| **Restrict connection** | select | None | Optionally require children to be "Touching parent" (intersecting) or "Within parent" (fully contained) |
| **Max distance (pixels)** | number | 1000 | Maximum allowed distance between child and parent (0-5000). Pairs beyond this distance are not connected |
| **Connect up to N children** | number | 10000 | Maximum number of children connected to each parent, selecting closest first (1-10000) |

## Implementation Details

### Distance Modes

- **Centroid mode**: Computes polygon centroids (or uses point coordinates directly), builds a `scipy.spatial.cKDTree`, and queries for the single nearest parent per child. This is fast but may not reflect true proximity for irregularly shaped polygons.
- **Edge mode**: Uses GeoPandas `sjoin_nearest` which measures distance between actual polygon boundaries. After the spatial join, duplicates are resolved by keeping the closest match per child.

### Connection Restrictions

When "Restrict connection" is set to "Touching parent" or "Within parent", the worker first filters children using Shapely geometric predicates (`intersects` or `within`) against the union of all parent geometries in the group. Only children passing this filter proceed to nearest-parent matching.

### Max Children per Parent

After computing all child-to-parent pairs, results are sorted by distance and grouped by parent. Only the closest N children (per the "Connect up to N children" parameter) are kept for each parent.

### Grouping Behavior

Annotations are always grouped by XY position (cross-XY connections are never created). Z and Time grouping can be individually disabled via "Connect across Z" and "Connect across T", which allows connecting objects across those dimensions.

### Output Tags

Created connections are tagged with the union of both parent and child tags.

## Notes

- Line annotations are not currently supported as input; only point and polygon annotations are used.
- For polygon annotations in centroid mode, the centroid of the polygon is used for distance calculation. In edge mode, the actual polygon geometry is used.
- The worker sends progress updates as it processes each location group.
