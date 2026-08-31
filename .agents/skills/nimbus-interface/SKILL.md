---
name: "nimbus-interface"
description: "Runtime NimbusImage/Girder API reference for worker code. Use for image loading, annotation CRUD, property values, channel merging, coordinate conversions, API test scripts, or infrastructure errors such as HTTP 500. Use nimbus-run-worker to execute a built image against a real local dataset, nimbus-worker-scaffold to create a worker, and nimbus-worker-hardening to diagnose or sweep failures."
---

# NimbusImage Worker Development

## Quick Start

Determine the task type:
- **Building/modifying a worker** → See [references/api.md](references/api.md) for full API patterns
- **Debugging HTTP 500 errors** → Check prerequisites below
- **Writing local test scripts** → See local testing section below
- **Coordinate confusion** → See critical pitfalls below

## Infrastructure Prerequisites

The Girder server requires **MongoDB**. Without it, all endpoints return HTTP 500 (except `/system/version`). Debug with:
```bash
docker ps | grep mongo  # Must be running
curl -s http://localhost:8080/api/v1/system/version  # Works without MongoDB
```

Full stack: `girder`, `worker` (celery), `rabbitmq`, `memcached`, `mongodb`.
The compose file lives in the separate NimbusImage checkout. Set
`NIMBUSIMAGE_REPO` to that checkout's root instead of assuming a personal path.

## Critical Pitfalls

### Coordinate swap (numpy vs annotations)
Numpy is `[row, col]` = `[y, x]`. Annotations use `{'x': pixel_x, 'y': pixel_y}`.
```python
# skimage contour (row, col) → annotation:
coords = [{'x': float(col), 'y': float(row)} for row, col in contour]

# Use annotation_tools helpers to avoid manual swaps:
from annotation_utilities.annotation_tools import polygons_to_annotations, annotations_to_polygons
```

### The 0.5 pixel offset
scikit-image uses pixel centers; Girder uses top-left corner:
```python
polygon = np.array([[c['y'] - 0.5, c['x'] - 0.5] for c in annotation['coordinates']])
rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1], shape=image.shape)
```

### Tags interface returns a list, not a dict
```python
# CORRECT:
tags = params['workerInterface'].get('Training Tag', [])
# WRONG (crashes with AttributeError):
tags = params['workerInterface'].get('Training Tag', {}).get('tags', [])
```

### Batch ranges are dataset-aware
Standard `Batch XY`, `Batch Z`, and `Batch Time` text fields accept 1-indexed
numeric ranges such as `1-3, 5-8`, case-insensitive `all`, or an empty string.
`all` expands from the dataset's `IndexRange`; empty fields use the current tile.
Use `WorkerClient` or `batch_argument_parser.get_batch_ranges(...)` so missing
dimensions safely fall back to coordinate `0`. See the detailed API reference
for direct-loop and custom-range examples.

### Multi-channel merge output dtype
`process_and_merge_channels` returns `float64` with values 0-255 (not 0-1). Convert for ML:
```python
rgb_uint8 = np.clip(merged, 0, 255).astype(np.uint8)
```

Typical shapes:
- `getRegion().squeeze()`: `(H, W)` uint16
- `get_images_for_all_channels`: each `(H, W, 1)` uint16
- `process_and_merge_channels`: `(H, W, 3)` float64, values 0-255

## Local Testing

To execute a built production image end to end against a live local dataset,
including safe authentication and output verification, use the
`nimbus-run-worker` skill. This section covers lower-level API test scripts.

### Avoid importing entrypoint.py
Worker entrypoints import heavy ML libraries (torch, sam2) at module level. Copy helper functions locally instead of importing the entrypoint.

### Local venv dependencies
```bash
# Run from the ImageAnalysisProject repository root.
export NIMBUSIMAGE_REPO="${NIMBUSIMAGE_REPO:-../NimbusImage}"
pip install girder-client tifffile
pip install -e "$NIMBUSIMAGE_REPO/devops/girder/annotation_client"
pip install -e ./annotation_utilities
pip install -e ./worker_client
pip install numpy scipy scikit-image shapely matplotlib pillow numba
# ML deps (torch, sam2, etc.) only needed for inference, not API testing
```

### Authentication for test scripts
```python
import girder_client
gc = girder_client.GirderClient(apiUrl='http://localhost:8080/api/v1')
gc.authenticate('username', 'password')
token = gc.token  # Use this token with annotation_client classes
```
Env vars: `NIMBUS_API_URL` (default `http://localhost:8080/api/v1`), `NIMBUS_TOKEN`.

### Test dataset
Dataset `69988c84b48d8121b565aba4`: 2 channels (Brightfield, YFP), 7Z, 4T, 6XY, 1024x1022 uint16. 544 polygons tagged "YFP blob" at XY=0 Z=3 Time=0.

## Key Packages

| Package | Location |
|---------|----------|
| annotation_client | `$NIMBUSIMAGE_REPO/devops/girder/annotation_client/` |
| annotation_utilities | `./annotation_utilities/` |
| worker_client | `./worker_client/` |
| Workers | `./workers/` |

Key source files: `annotation_client/{annotations,tiles,workers}.py`, `annotation_utilities/{annotation_tools,batch_argument_parser}.py`

## Detailed API Reference

See [references/api.md](references/api.md) for complete API patterns including:
- Image access (single frame, subregion, multi-channel merge)
- Annotation CRUD (fetch, filter, create, delete)
- Property value computation and submission
- Writing images back to Girder
- Worker interface type table
- Dataset-aware Batch XY/Z/Time parsing, including `all`
