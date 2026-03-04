# Blob Metrics

Computes geometric and shape metrics for polygon annotations using Shapely. Optionally reports values in physical units.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Use physical units** | checkbox | false | If checked, area and perimeter are reported in physical units instead of pixels. |
| **Units** | select | um | Target physical units: m, mm, um, nm. Only used when physical units are enabled. |

## Computed Properties

| Property | Description |
|----------|-------------|
| Area | Polygon area (px^2 or physical units^2) |
| Perimeter | Polygon perimeter length (px or physical units) |
| Centroid | Centroid coordinates as `{x, y}` |
| Elongation | 1 - (min bounding rect width / length). 0 = circle-like, 1 = line-like. |
| Convexity | Ratio of polygon area to convex hull area |
| Solidity | Ratio of polygon perimeter to convex hull perimeter |
| Rectangularity | Ratio of polygon area to minimum bounding rectangle area |
| Circularity | 4*pi*area / perimeter^2. 1 = perfect circle. |
| Eccentricity | Derived from eigenvalues of the inertia tensor. 0 = circle, approaching 1 = elongated. |

## How It Works

1. Each annotation's coordinates are converted to a Shapely `Polygon`.
2. The minimum rotated rectangle is computed for elongation (via edge lengths of the bounding rectangle).
3. Eccentricity is calculated from eigenvalues of the inertia tensor of centered polygon coordinates.
4. All ratio-based metrics use a safe division wrapper that returns `None` for degenerate cases.

## Notes

- Polygons with fewer than 3 points are skipped with a warning.
- Physical unit conversion uses the dataset's `pixelSize` from `params['scales']`. If pixel size is 0, physical units are disabled automatically.
- Centroid values are also scaled by the pixel length when physical units are enabled.
- Coordinate values use Shapely's convention (x, y) directly from annotation coordinates, without the 0.5-pixel offset used for raster operations.
