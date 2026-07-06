# Ultrack Worker

This worker links **existing** segmentation objects across time into tracks using [Ultrack](https://github.com/royerlab/ultrack), which solves a global optimization over the segmentations rather than linking frame-by-frame. It does not create new segmentations — you segment your objects first (e.g. with Cellpose, Cellpose-SAM, or Stardist), tag them, then run this tool to build the lineage as parent-child connections. Cell divisions are captured as one parent linked to two daughters.

## How It Works

1. **Load objects**: Fetches all polygon annotations carrying the selected tag.
2. **Group by position**: Objects are grouped by XY position and Z slice; each stack is tracked independently across time.
3. **Rasterize to label masks**: For each time point, the tagged polygons are drawn into an integer label image, giving a `(T, H, W)` mask stack. The centroid of each label and its `(label, time) → annotation id` mapping are retained.
4. **Track**: Ultrack ingests the label stack, derives detection/contour hypotheses, and solves a global integer linear program to produce a tracks table with `track_id`, `t`, `y`, `x`, and `parent_track_id`.
5. **Map back and connect**: Each track node is matched to the nearest original annotation (by mask centroid at that time point). Consecutive nodes within a track become motion connections, and each `parent_track_id` produces a division connection from the parent track's last node to the child track's first node.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Tag of objects to track** | tags | -- | Tag identifying which polygon annotations to track |
| **Max distance** | number (pixels) | 50 | Maximum distance an object may move between consecutive time points to be linked |
| **Allow divisions** | checkbox | True | Allow one object to split into two (cell division). Uncheck to forbid divisions |
| **Batch XY** | text | current XY | XY positions to track, e.g. `1-3, 5`. Blank = current position |
| **Batch Z** | text | current Z | Z slices to track, e.g. `1-3, 5`. Blank = current slice |

## Implementation Details

### Coordinate Handling

Polygon coordinates are converted from Girder's top-left-origin convention to scikit-image pixel centers (the 0.5 offset) when rasterizing. Rows correspond to `y`, columns to `x`. The tracking canvas uses the dataset's full image dimensions (`sizeX`, `sizeY`).

### Mapping Tracks Back to Annotations

Ultrack relabels objects internally, so track nodes are matched back to the original annotations by nearest mask centroid at the same time point (both are expressed in `(row, col)` = `(y, x)` image space, so the match is exact for undivided objects). This keeps the worker independent of Ultrack's internal id scheme.

### Divisions

Ultrack encodes lineage via `parent_track_id`. When a track has a parent, a connection is created from the parent track's final node to that child track's first node — so a division yields two such connections. Unchecking **Allow divisions** applies a strong penalty to division events in the solver.

### Working Directory

Ultrack persists intermediate results to a SQLite database; each XY/Z stack is tracked inside its own temporary directory that is cleaned up afterwards.

## Notes

- **CPU**: Ultrack's tracking is CPU-based and uses the open-source COIN-OR CBC solver by default (Gurobi is optional and not required).
- Only polygon annotations are tracked. Objects at a single time point (no second frame in their XY/Z stack) cannot form a track and are skipped.
- Related workers: **Trackastra** (transformer-based tracking of the same segmentations), **Connect Time Lapse** and **Connect Sequential** (simpler nearest-neighbor linking), **SAM2 video** (segment-and-track in one step).
