# Trackastra Worker

This worker links **existing** segmentation objects across time into tracks using [Trackastra](https://github.com/weigertlab/trackastra), a transformer-based cell tracking model. It does not create new segmentations — you segment your objects first (e.g. with Cellpose, Cellpose-SAM, or Stardist), tag them, then run this tool to build the lineage as parent-child connections. Cell divisions are captured as one parent linked to two daughters.

## How It Works

1. **Load objects**: Fetches all polygon annotations carrying the selected tag.
2. **Group by position**: Objects are grouped by XY position and Z slice; each stack is tracked independently across time.
3. **Rasterize to label masks**: For each time point, the tagged polygons are drawn into an integer label image, giving a `(T, H, W)` mask stack. A `(label, time) → annotation id` map is retained.
4. **Load intensity images**: The raw image for the selected channel is loaded for each time point, giving a matching `(T, H, W)` image stack.
5. **Track**: The pretrained Trackastra model links detections across frames, producing a directed track graph (a node per detection, edges from a detection to its successor, and two out-edges where a cell divides).
6. **Create connections**: Each graph edge is mapped back to the original annotations and uploaded as a parent-child connection.

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Tag of objects to track** | tags | -- | Tag identifying which polygon annotations to track |
| **Channel** | channel | -- | Intensity channel the model looks at (should match the channel the objects were segmented on) |
| **Model** | select | general_2d | Pretrained Trackastra model |
| **Tracking mode** | select | greedy | `greedy` (allows divisions) or `greedy_nodiv` (no divisions) |
| **Batch XY** | text | current XY | XY positions to track, e.g. `1-3, 5`. Blank = current position |
| **Batch Z** | text | current Z | Z slices to track, e.g. `1-3, 5`. Blank = current slice |

## Implementation Details

### Coordinate Handling

Polygon coordinates are converted from Girder's top-left-origin convention to scikit-image pixel centers (the 0.5 offset) when rasterizing. Rows correspond to `y`, columns to `x`. The tracking canvas uses the dataset's full image dimensions (`sizeX`, `sizeY`).

### Mapping Tracks Back to Annotations

Each mask label is unique within its frame and recorded against the annotation it came from. Trackastra graph nodes carry `time` (frame index) and `label`, so every node maps directly back to an annotation. Connections are oriented parent → child by time (not by stored edge direction), and duplicate parent/child pairs are de-duplicated.

### Divisions

In `greedy` mode a dividing cell produces a node with two outgoing edges, which becomes two connections (parent → daughter A, parent → daughter B). Use `greedy_nodiv` to forbid divisions.

### Output Tags

Created connections always carry a `Trackastra` tag (in addition to any output tags configured on the tool) so the tracking result can be filtered, selected, or bulk-deleted as a group in the UI.

## Notes

- **GPU**: Uses PyTorch and runs on GPU when available, falling back to CPU otherwise.
- **Model weights** are pre-downloaded into the image (best-effort at build time; otherwise fetched on first run).
- Trackastra also ships an `ilp` linking mode, but it requires an ILP solver (`ilpy` + SCIP/Gurobi) that is not installed in this image, so only the greedy modes are exposed.
- Only polygon annotations are tracked, and each frame is rasterized into a single label image, so objects are expected to be **non-overlapping** instance segmentations. Where two tagged polygons overlap at the same XY/Z/Time, the later-drawn one wins and a fully occluded object is dropped from tracking. Polygons that fall entirely outside the image bounds rasterize to nothing and are likewise skipped.
- Objects at a single time point (no second frame in their XY/Z stack) cannot form a track and are skipped.
- Related workers: **Ultrack** (optimization-based tracking of the same segmentations), **Connect Time Lapse** and **Connect Sequential** (simpler nearest-neighbor linking), **SAM2 video** (segment-and-track in one step).
