# Blob Overlap

Computes the fractional area overlap between two sets of polygon annotations, identified by their tags. Uses GeoPandas spatial overlay for efficient intersection computation.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Annotations to compute overlap with** | tags | (required) | Tags identifying the second set of annotations to compute overlap against. |
| **Compute reverse overlaps** | checkbox | true | If checked, also computes overlap from the second set's perspective and stores it on those annotations. |

## Computed Properties

| Property | Description |
|----------|-------------|
| Overlap_{tags} | Fraction of annotation area overlapping with the other set (intersection area / annotation area). Tag names are joined with underscores. |

When reverse overlaps are enabled, both sets of annotations receive overlap properties for both tag groups (their own overlap set to 0.0 as a placeholder).

## How It Works

1. The property's own tags select the first annotation set; the "Annotations to compute overlap with" tags select the second set.
2. Both sets are converted to GeoPandas GeoDataFrames with Shapely polygon geometries.
3. Annotations are grouped by (Time, XY, Z) location. Only annotations at the same location are compared.
4. For each location group, `gpd.overlay(..., how='intersection')` computes all pairwise polygon intersections.
5. For each annotation, the total intersection area is divided by the annotation's own area to get the overlap fraction.
6. If reverse overlaps are enabled, the same process runs from the second set's perspective.

## Notes

- Polygons with fewer than 3 coordinates are skipped during GeoDataFrame construction.
- Annotations at different (Time, XY, Z) locations are never compared against each other.
- The overlap value can exceed 1.0 if an annotation overlaps with multiple annotations from the other set (the intersections are summed).
