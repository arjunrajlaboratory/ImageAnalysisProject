# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains Docker-based workers for NimbusImage, a cloud platform for image analysis. Workers interface with a Girder/large_image backend to pull images and annotations, then return annotations or computed property values to the server.

## Build Commands

```bash
# Build all workers (auto-detects arm64/x86_64 architecture)
./build_workers.sh

# Build a specific worker
./build_workers.sh blob_metrics

# Build without cache
./build_workers.sh --no-cache

# Build and run all tests
./build_workers.sh --build-and-run-tests

# Build and run tests for a specific worker
./build_workers.sh --build-and-run-tests blob_metrics

# Build tests only (no run)
./build_workers.sh --build-tests-only

# Run tests only (no build)
./build_workers.sh --run-tests-only

# Build GPU/ML workers (cellpose, SAM2, piscis, stardist, condensatenet)
./build_machine_learning_workers.sh

# Mac development mode (forces CPU-only Dockerfile_M1 for workers with GPU defaults)
MAC_DEVELOPMENT_MODE=true ./build_workers.sh deconwolf
```

## Architecture

### Worker Types

Workers live in `workers/` and are organized into two categories:

- **`workers/annotations/`**: Create annotations via segmentation (cellposesam, histogram_matching, connect_to_nearest, etc.)
- **`workers/properties/`**: Calculate properties on existing annotations, organized by shape:
  - `blobs/`: Polygon/blob properties (intensity, metrics, overlap)
  - `points/`: Point annotation properties (intensity, distance)
  - `lines/`: Line annotation properties (length, scan)
  - `connections/`: Relationship properties (children_count, parent_child)

### Worker Entry Point Pattern

Every worker has an `entrypoint.py` with two required functions:

```python
def interface(image, apiUrl, token):
    """Define the UI interface shown to users"""
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    interface = {
        'Channel': {'type': 'channel', 'required': True, ...},
        'Radius': {'type': 'number', 'min': 0, 'max': 10, ...},
    }
    client.setWorkerImageInterface(image, interface)

def compute(datasetId, apiUrl, token, params):
    """Main computation logic"""
    # params contains: workerInterface, tags, tile, assignment, channel, connectTo
    ...
```

Interface types: `number`, `text`, `select`, `checkbox`, `channel`, `channelCheckboxes`, `tags`, `layer`, `notes`

### Interface Parameter Data Types (What `params['workerInterface']` Returns)

Each interface type returns a specific data type in `params['workerInterface']['FieldName']`:

| Interface Type | Returns | Example Value |
|----------------|---------|---------------|
| `number` | `int` or `float` | `32`, `0.5` |
| `text` | `str` | `"1-3, 5-8"`, `""` |
| `select` | `str` | `"sam2.1_hiera_small.pt"` |
| `checkbox` | `bool` | `True`, `False` |
| `channel` | `int` | `0` |
| `channelCheckboxes` | `dict` of `str` → `bool` | `{"0": True, "1": False, "2": True}` |
| `tags` | **`list` of `str`** | `["DAPI blob"]`, `["cell", "nucleus"]` |
| `layer` | `str` | `"layer_id"` |

**Common pitfall with `tags`**: The `tags` type returns a **plain list of strings**, NOT a dict. Do not call `.get('tags')` on the result.

```python
# CORRECT - tags returns a list directly:
training_tags = params['workerInterface'].get('Training Tag', [])
# training_tags = ["DAPI blob"]

# WRONG - will crash with AttributeError: 'list' object has no attribute 'get':
training_tags = params['workerInterface'].get('Training Tag', {}).get('tags', [])
```

**Note**: `params['tags']` (the top-level output tags for the worker, NOT a workerInterface field) is also a plain list of strings (e.g., `["DAPI blob"]`). Meanwhile, `params['tags']` used in property workers via `workerClient.get_annotation_list_by_shape()` uses `{'tags': [...], 'exclusive': bool}` — these are two different things.

**Validating tags** (recommended pattern from cellpose_train, piscis):
```python
tags = workerInterface.get('My Tag Field', [])
if not tags or len(tags) == 0:
    sendError("No tag selected", "Please select at least one tag.")
    return
```

**Using tags to filter annotations**:
```python
# Pass the list directly to annotation_tools
filtered = annotation_tools.get_annotations_with_tags(
    annotation_list, tags, exclusive=False)

# Or with Girder API (must JSON-serialize)
annotations = annotationClient.getAnnotationsByDatasetId(
    datasetId, shape='polygon', tags=json.dumps(tags))
```

### Key APIs

**annotation_client** (installed from NimbusImage repo):
- `annotation_client.workers`: `UPennContrastWorkerClient`, `UPennContrastWorkerPreviewClient`
- `annotation_client.tiles`: `UPennContrastDataset` - access images via large_image
- `annotation_client.annotations`: `UPennContrastAnnotationClient` - CRUD for annotations
- `annotation_client.utils`: `sendProgress(fraction, title, info)`, `sendWarning(msg, info)`, `sendError(msg, info)`

**annotation_utilities** (local package in `annotation_utilities/`):
- `annotation_tools`: Filter annotations by tags, convert to/from shapely geometries
- `batch_argument_parser`: Parse batch ranges like "1-3, 5-8"
- `progress`: `update_progress()` helper

**worker_client** (local package in `worker_client/`):
- `WorkerClient`: High-level class for annotation workers that handles batching over XY/Z/Time

### Coordinate Conventions (Critical)

There are important coordinate transformations between scikit-image/numpy and Girder/large_image annotations:

**1. The 0.5 pixel offset**: scikit-image uses pixel centers, while Girder uses top-left corner as origin. When reading annotation coordinates for use with scikit-image functions like `draw.polygon()`:

```python
# Reading annotation coordinates for scikit-image processing
polygon = np.array([[coord['y'] - 0.5, coord['x'] - 0.5]
                    for coord in annotation['coordinates']])
rr, cc = draw.polygon(polygon[:, 0], polygon[:, 1], shape=image.shape)
```

**2. The x/y swap**: Numpy arrays are indexed as `[row, col]` which corresponds to `[y, x]`. Annotation coordinates use `{'x': ..., 'y': ...}`. The conversion functions in `annotation_tools.py` handle this:

```python
# annotation_tools.annotations_to_points(): annotation coords → shapely Point
y, x = coords['x'], coords['y']  # Note the swap
point = Point(x, y)

# annotation_tools.polygons_to_annotations(): shapely Polygon → annotation coords
coordinates = [{'x': float(y), 'y': float(x)} for x, y in polygon.exterior.coords]
```

**3. Helper functions** in `annotation_utilities.annotation_tools`:
- `annotations_to_polygons()` / `polygons_to_annotations()` - handle coordinate swaps for polygons
- `annotations_to_points()` / `points_to_annotations()` - handle coordinate swaps for points

**When creating annotations directly** (like in stardist), if your polygon coordinates come from rasterio/shapely operations that already match image coordinates, you can use them directly:
```python
# If coords are already in (x, y) image space from rasterio.features.shapes()
"coordinates": [{"x": float(x), "y": float(y)} for x, y in polygon.exterior.coords]
```

### Image Access Pattern

```python
# Get tile client for image access
tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

# Convert coordinates to frame index
frame = tileClient.coordinatesToFrameIndex(xy, z, time, channel)

# Get image region as numpy array
image = tileClient.getRegion(datasetId, frame=frame).squeeze()

# Get a subregion (useful for cropping to ROI)
image = tileClient.getRegion(datasetId, frame=frame,
                              left=x_min, top=y_min, right=x_max, bottom=y_max,
                              units="base_pixels").squeeze()

# Access tile metadata
num_channels = tileClient.tiles['IndexRange'].get('IndexC', 1)
num_z = tileClient.tiles['IndexRange'].get('IndexZ', 1)
num_time = tileClient.tiles['IndexRange'].get('IndexT', 1)
num_xy = tileClient.tiles['IndexRange'].get('IndexXY', 1)
```

### Image Processing Workflow (Writing New Images to Girder)

For workers that process images and upload results back to Girder (like histogram_matching, registration):

```python
import large_image as li

# Create a new image sink
sink = li.new()

# Process each frame and add to sink
for i, frame in enumerate(tileClient.tiles['frames']):
    # Build large_image params from frame indices
    large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items()
                          if k.startswith('Index') and len(k) > 5}

    image = tileClient.getRegion(datasetId, frame=i).squeeze()
    processed_image = your_processing_function(image)

    sink.addTile(processed_image, 0, 0, **large_image_params)
    sendProgress(i / len(tileClient.tiles['frames']), 'Processing', f"Frame {i+1}")

# Copy metadata from source
if 'channels' in tileClient.tiles:
    sink.channelNames = tileClient.tiles['channels']
sink.mm_x = tileClient.tiles['mm_x']
sink.mm_y = tileClient.tiles['mm_y']
sink.magnification = tileClient.tiles['magnification']

# Write to temp file and upload
sink.write('/tmp/output.tiff')
gc = tileClient.client
item = gc.uploadFileToFolder(datasetId, '/tmp/output.tiff')
gc.addMetadataToItem(item['itemId'], {'tool': 'YourWorkerName', ...})
```

### Annotation Creation Workflow

For workers that create annotations (like stardist, cellposesam):

```python
annotationClient = annotations.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)

# Build annotation objects
out_annotations = []
for polygon in detected_polygons:
    annotation = {
        "tags": params.get('tags', []),
        "shape": "polygon",  # or "point", "line"
        "channel": channel,
        "location": {"XY": tile['XY'], "Z": tile['Z'], "Time": tile['Time']},
        "datasetId": datasetId,
        "coordinates": [{"x": float(x), "y": float(y)} for x, y in polygon.exterior.coords],
    }
    out_annotations.append(annotation)

# Upload all at once
annotationClient.createMultipleAnnotations(out_annotations)
```

**Using WorkerClient for batched annotation creation** (handles XY/Z/Time iteration):
```python
from worker_client import WorkerClient

worker = WorkerClient(datasetId, apiUrl, token, params)

def process_image(image):
    # Return list of polygon coordinates as [(x,y), (x,y), ...]
    return detected_polygons

# Automatically iterates over Batch XY/Z/Time from interface
worker.process(process_image, f_annotation='polygon', stack_channels=[channel])
```

### Property Value Creation Workflow

For workers that compute properties on existing annotations (like blob_metrics, blob_intensity):

```python
workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)

# Get annotations filtered by shape and tags
annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
annotationList = annotation_tools.get_annotations_with_tags(
    annotationList,
    params.get('tags', {}).get('tags', []),
    params.get('tags', {}).get('exclusive', False)
)

# Build property values dictionary: {annotation_id: {prop_name: value, ...}}
property_value_dict = {}
for annotation in annotationList:
    prop = {
        'Area': float(computed_area),
        'Perimeter': float(computed_perimeter),
        'MeanIntensity': float(mean_val),
        # ... more properties
    }
    property_value_dict[annotation['_id']] = prop

# Wrap with datasetId and send to server
dataset_property_value_dict = {datasetId: property_value_dict}
workerClient.add_multiple_annotation_property_values(dataset_property_value_dict)
```

**For nested/multi-dimensional properties** (e.g., intensity across multiple Z planes):
```python
# Initialize nested structure for each annotation
for annotation in annotationList:
    property_value_dict[annotation['_id']] = {
        'MeanIntensity': {},  # Will hold {z_key: value, ...}
        'MaxIntensity': {},
    }

# Fill in values for each Z plane
for z in z_planes:
    z_key = f"z{(z+1):03d}"  # e.g., "z001", "z002"
    for annotation in annotations_at_location:
        # ... compute intensities ...
        property_value_dict[annotation['_id']]['MeanIntensity'][z_key] = float(mean_val)
        property_value_dict[annotation['_id']]['MaxIntensity'][z_key] = float(max_val)
```

**Accessing pixel scale from params** (for physical units):
```python
pixelSize = params['scales']['pixelSize']  # {'unit': 'mm', 'value': 0.000219}
# params also has: params['scales']['tStep'], params['scales']['zStep']
```

### Docker Structure

- Base images defined in `workers/base_docker_images/`
- Workers inherit from `nimbusimage/worker-base:latest` or `nimbusimage/image-processing-base:latest`
- Docker labels identify worker type: `isPropertyWorker`, `isAnnotationWorker`, `annotationShape`, `interfaceName`, `interfaceCategory`
- Architecture-aware builds: `Dockerfile` (x86_64/production) and `Dockerfile_M1` (arm64/Mac development)
- GPU workers (deconwolf, condensatenet, etc.) use NVIDIA CUDA base images by default
  - Set `MAC_DEVELOPMENT_MODE=true` to build CPU-only versions on Mac
  - GPU workers have automatic CPU fallback at runtime if OpenCL/CUDA unavailable

### Testing

Tests use pytest with mocked `annotation_client`. Test files go in `tests/` subdirectory:
```
workers/properties/blobs/blob_intensity_worker/
├── entrypoint.py
├── Dockerfile
└── tests/
    ├── __init__.py
    ├── test_blob_intensity.py
    └── Dockerfile_Test
```

## Example Workers to Reference

When creating new workers, use these as templates:

- **Property worker (blobs)**: `workers/properties/blobs/blob_intensity_worker/entrypoint.py`
- **Property worker (points)**: `workers/properties/points/point_circle_intensity_worker/entrypoint.py`
- **Annotation worker (ML)**: `workers/annotations/cellposesam/entrypoint.py`
- **Image processing worker**: `workers/annotations/histogram_matching/entrypoint.py`
- **Image processing worker (GPU)**: `workers/annotations/deconwolf/entrypoint.py` - GPU-accelerated with CPU fallback
- **Sample/test interface**: `workers/properties/blobs/sample_interface_worker/entrypoint.py`
