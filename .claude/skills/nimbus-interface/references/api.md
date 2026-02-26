# NimbusImage API Reference

## Table of Contents
- [Image Access](#image-access)
- [Annotations](#annotations)
- [Property Values](#property-values)
- [Writing Images to Girder](#writing-images-to-girder)
- [Worker Interface Types](#worker-interface-types)

---

## Image Access

### Setup
```python
import annotation_client.tiles as tiles
tileClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)
```

### Metadata
```python
idx = tileClient.tiles['IndexRange']
num_channels = idx.get('IndexC', 1)
num_z = idx.get('IndexZ', 1)
num_time = idx.get('IndexT', 1)
num_xy = idx.get('IndexXY', 1)
size_x = tileClient.tiles['sizeX']
size_y = tileClient.tiles['sizeY']
channel_names = tileClient.tiles.get('channels', [])
pixel_scale = tileClient.tiles.get('mm_x')  # mm per pixel
```

### Single frame
```python
frame = tileClient.coordinatesToFrameIndex(XY, Z=z, T=time, channel=channel)
image = tileClient.getRegion(datasetId, frame=frame).squeeze()
# Returns (H, W) uint16
```

### Subregion
```python
image = tileClient.getRegion(datasetId, frame=frame,
    left=x_min, top=y_min, right=x_max, bottom=y_max,
    units="base_pixels").squeeze()
```

### Multi-channel merged RGB
```python
import annotation_utilities.annotation_tools as annotation_tools

images = annotation_tools.get_images_for_all_channels(tileClient, datasetId, XY, Z, Time)
# Each: (H, W, 1) uint16
layers = annotation_tools.get_layers(tileClient.client, datasetId)
merged = annotation_tools.process_and_merge_channels(images, layers)
# Returns: (H, W, 3) float64, values 0-255
```
Merge modes: `'lighten'` (max, default), `'add'` (sum), `'screen'`.

---

## Annotations

### Client setup
```python
import annotation_client.annotations as annotations_client
annotationClient = annotations_client.UPennContrastAnnotationClient(apiUrl=apiUrl, token=token)
```

### Data structure
```python
{
    'shape': 'polygon',  # or 'point', 'line'
    'coordinates': [{'x': float, 'y': float}, ...],
    'location': {'XY': int, 'Z': int, 'Time': int},
    'channel': int,
    'datasetId': str,
    'tags': ['tag1', 'tag2'],
}
```

### Fetch
```python
polygons = annotationClient.getAnnotationsByDatasetId(datasetId, shape='polygon')

# Filter by tags server-side (must JSON-serialize)
import json
polygons = annotationClient.getAnnotationsByDatasetId(
    datasetId, shape='polygon', tags=json.dumps(['my_tag']))

ann = annotationClient.getAnnotationById(annotationId)
```

### Client-side filtering
```python
import annotation_utilities.annotation_tools as annotation_tools

filtered = annotation_tools.get_annotations_with_tags(annotations, tags, exclusive=False)
# exclusive=False: any matching tag; exclusive=True: exact tag set match

filtered = annotation_tools.filter_elements_T_XY_Z(annotations, time, xy, z)
```

### Create
```python
annotationClient.createAnnotation(annotation_dict)
annotationClient.createMultipleAnnotations(annotation_list)  # preferred

# Using helpers (handles coordinate swap):
from annotation_utilities.annotation_tools import polygons_to_annotations
annotations = polygons_to_annotations(
    shapely_polygons, datasetId, XY=0, Time=0, Z=0, tags=['my_tag'], channel=0)
```

### Delete
```python
annotationClient.deleteAnnotation(annotationId)
annotationClient.deleteMultipleAnnotations([id1, id2, ...])
```

---

## Property Values

### Setup
```python
import annotation_client.workers as workers
workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
```

### Get annotations for computation
```python
annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
annotationList = annotation_tools.get_annotations_with_tags(
    annotationList,
    params.get('tags', {}).get('tags', []),
    params.get('tags', {}).get('exclusive', False))
```

### Submit values
```python
property_values = {}
for ann in annotationList:
    property_values[ann['_id']] = {
        'Area': float(area),
        'MeanIntensity': float(mean),
    }
workerClient.add_multiple_annotation_property_values({datasetId: property_values})
```

### Nested properties (per-Z, per-channel)
```python
property_values[ann['_id']] = {
    'MeanIntensity': {'z001': 42.0, 'z002': 84.0},
}
```

### Pixel scale
```python
pixel_size = params['scales']['pixelSize']  # {'unit': 'mm', 'value': 0.000219}
z_step = params['scales']['zStep']
t_step = params['scales']['tStep']
```

---

## Writing Images to Girder

```python
import large_image as li

sink = li.new()
for i, frame in enumerate(tileClient.tiles['frames']):
    large_image_params = {f'{k.lower()[5:]}': v for k, v in frame.items()
                          if k.startswith('Index') and len(k) > 5}
    image = tileClient.getRegion(datasetId, frame=i).squeeze()
    processed = your_function(image)
    sink.addTile(processed, 0, 0, **large_image_params)

if 'channels' in tileClient.tiles:
    sink.channelNames = tileClient.tiles['channels']
sink.mm_x = tileClient.tiles['mm_x']
sink.mm_y = tileClient.tiles['mm_y']
sink.magnification = tileClient.tiles['magnification']

sink.write('/tmp/output.tiff')
gc = tileClient.client
item = gc.uploadFileToFolder(datasetId, '/tmp/output.tiff')
gc.addMetadataToItem(item['itemId'], {'tool': 'YourWorker'})
```

---

## Worker Interface Types

| Type | Returns | Example |
|------|---------|---------|
| `number` | `int`/`float` | `32`, `0.5` |
| `text` | `str` | `"1-3, 5-8"` |
| `select` | `str` | `"model_name.pt"` |
| `checkbox` | `bool` | `True` |
| `channel` | `int` | `0` |
| `channelCheckboxes` | `dict[str, bool]` | `{"0": True, "1": False}` |
| `tags` | `list[str]` | `["DAPI blob"]` |
| `layer` | `str` | `"layer_id"` |
