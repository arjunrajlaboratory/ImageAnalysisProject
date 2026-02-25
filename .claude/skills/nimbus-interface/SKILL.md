---
name: nimbus-interface
description: Reference for the NimbusImage/Girder API used by all workers in this repository. Use when building, debugging, or testing NimbusImage workers — including image loading, annotation CRUD, property computation, multi-channel merging, coordinate conversions, local test environments, and infrastructure troubleshooting (e.g. HTTP 500 errors). Also use when writing test scripts that interact with the Nimbus API.
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
Compose file: `/home/arjun/UPennContrast/docker-compose.yaml`.

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

### Avoid importing entrypoint.py
Worker entrypoints import heavy ML libraries (torch, sam2) at module level. Copy helper functions locally instead of importing the entrypoint.

### Local venv dependencies
```bash
pip install girder-client tifffile
pip install -e /home/arjun/UPennContrast/devops/girder/annotation_client
pip install -e /home/arjun/ImageAnalysisProject/annotation_utilities
pip install -e /home/arjun/ImageAnalysisProject/worker_client
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
| annotation_client | `/home/arjun/UPennContrast/devops/girder/annotation_client/` |
| annotation_utilities | `/home/arjun/ImageAnalysisProject/annotation_utilities/` |
| worker_client | `/home/arjun/ImageAnalysisProject/worker_client/` |
| Workers | `/home/arjun/ImageAnalysisProject/workers/` |

Key source files: `annotation_client/{annotations,tiles,workers}.py`, `annotation_utilities/{annotation_tools,batch_argument_parser}.py`

## Detailed API Reference

See [references/api.md](references/api.md) for complete API patterns including:
- Image access (single frame, subregion, multi-channel merge)
- Annotation CRUD (fetch, filter, create, delete)
- Property value computation and submission
- Writing images back to Girder
- Worker interface type table
