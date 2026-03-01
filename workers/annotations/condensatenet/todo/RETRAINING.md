# CondensateNet Retraining: NimbusImage Worker

## Overview

Add a training worker for CondensateNet in NimbusImage, and update the existing prediction worker to support selecting custom models. This is the **second step** — the condensatenet Python package must first have its training API implemented (see the companion `RETRAINING.md` in the `condensatenet` repo at `todo/RETRAINING.md`).

## Prerequisites

The `condensatenet` package (github.com/arjunrajlaboratory/condensatenet) must expose:
- `condensatenet.data.prepare_training_data(images, label_masks, patch_size, augment, random_seed)` → `CondensateNetDataset`
- `condensatenet.training.train_model(dataset, initial_model_path, output_path, epochs, learning_rate, ..., progress_callback)` → saved model path
- Trained models saved in HuggingFace format (`config.json` + `model.safetensors`), loadable by existing `CondensateNetPipeline.from_local()`

## Architecture: Piscis-Style train/predict Split

Restructure the condensatenet worker from a single container to a train/predict split with shared utilities, following the piscis worker pattern (`workers/annotations/piscis/`).

**Current structure:**
```
workers/annotations/condensatenet/
├── entrypoint.py
├── Dockerfile
├── environment.yml
└── CONDENSATENET.md
```

**New structure:**
```
workers/annotations/condensatenet/
├── docker-compose.yaml          # NEW — builds both containers
├── utils.py                     # NEW — Girder model management (shared)
├── environment.yml              # EXISTING (unchanged)
├── CONDENSATENET.md             # EXISTING (update with training docs)
├── todo/
│   └── RETRAINING.md            # This file (remove after implementation)
├── predict/
│   ├── Dockerfile               # MOVED+MODIFIED from current Dockerfile
│   └── entrypoint.py            # MOVED+MODIFIED from current entrypoint.py
└── train/
    ├── Dockerfile               # NEW
    └── entrypoint.py            # NEW
```

**Why piscis-style (not cellpose-style)?** Cellpose uses separate directories (`cellpose_train/` vs `cellposesam/`) because they share nothing beyond the pip package. CondensateNet train and predict need shared `utils.py` for Girder model management, shared `environment.yml`, and a single `docker-compose.yaml`. The piscis pattern handles all of this cleanly.

---

## Implementation Details

### 1. `utils.py` — Girder Model Management

Follows the pattern from `workers/annotations/piscis/utils.py` and `workers/annotations/cellpose_train/girder_utils.py`, adapted for CondensateNet's directory-based model format.

**Key difference from piscis**: Piscis models are single `.pt` files. CondensateNet models are directories containing `config.json` + `model.safetensors` (HuggingFace format). We handle this by archiving as `.tar.gz` for Girder storage.

```python
"""Girder model management utilities for CondensateNet."""

from pathlib import Path
import tarfile
import tempfile

MODELS_DIR = Path('/root/.condensatenet/models')


def mkdir(gc, parent_id, folder_name):
    """Create a Girder folder if it doesn't exist, return its ID."""
    folders = gc.get('folder', parameters={
        'parentId': parent_id, 'parentType': 'folder', 'name': folder_name
    })
    if folders:
        return folders[0]['_id']
    new_folder = gc.post('folder', parameters={
        'parentId': parent_id, 'parentType': 'folder', 'name': folder_name
    })
    return new_folder['_id']


def get_condensatenet_dir(gc):
    """Get the Private/.condensatenet folder, creating if needed."""
    user_id = gc.get('user/me')['_id']
    private_folder_id = gc.get('folder', parameters={
        'parentId': user_id, 'parentType': 'user', 'name': 'Private'
    })[0]['_id']
    return mkdir(gc, private_folder_id, '.condensatenet')


def list_girder_models(gc):
    """List custom models stored in Girder. Returns (model_list, models_folder_id)."""
    condensatenet_folder_id = get_condensatenet_dir(gc)
    models_folder_id = mkdir(gc, condensatenet_folder_id, 'models')
    girder_models = list(gc.listItem(models_folder_id))
    for model in girder_models:
        # Strip .tar.gz extension to get model name
        model['model_name'] = model['name'].replace('.tar.gz', '')
    return girder_models, models_folder_id


def download_girder_model(gc, model_name):
    """Download and extract a model archive from Girder to MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    girder_models, _ = list_girder_models(gc)
    matching = [m for m in girder_models if m['model_name'] == model_name]
    if matching:
        archive_name = matching[0]['name']
        gc.downloadItem(matching[0]['_id'], str(MODELS_DIR), archive_name)
        archive_path = MODELS_DIR / archive_name
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(MODELS_DIR)
        archive_path.unlink()  # Clean up archive


def upload_girder_model(gc, model_name, model_dir):
    """Archive a model directory and upload to Girder."""
    girder_models, models_folder_id = list_girder_models(gc)

    # Delete existing model with same name
    matching = [m for m in girder_models if m['model_name'] == model_name]
    if matching:
        gc.delete(f"{matching[0]['_modelType']}/{matching[0]['_id']}")

    # Create tar.gz archive
    archive_path = Path(tempfile.mkdtemp()) / f'{model_name}.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(model_dir, arcname=model_name)

    gc.uploadFileToFolder(models_folder_id, str(archive_path))
```

**Girder folder structure** (per user):
```
Private/
└── .condensatenet/
    └── models/
        ├── my_custom_model.tar.gz      (contains config.json + model.safetensors)
        └── retrained_20260228.tar.gz
```

---

### 2. Training Worker — `train/entrypoint.py`

#### Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| CondensateNet Train | notes | — | Informational text about the training workflow |
| Initial Model | select | pretrained | Base model to fine-tune (pretrained + Girder models listed dynamically) |
| New Model Name | text | timestamp | Name for the trained model (e.g., `20260228_143022`) |
| Annotation Tag | tags | — | **Required.** Tag(s) identifying ground-truth polygon annotations (corrected condensate boundaries) |
| Region Tag | tags | — | **Required.** Tag(s) identifying training regions (polygons/rectangles enclosing fully-annotated areas) |
| Channel | channel | — | Image channel to train on |
| Epochs | number | 100 | Number of training epochs (min 10, max 500) |
| Learning Rate | text | 0.0001 | Fine-tuning learning rate |
| Patch Size | number | 256 | Random crop size for training patches (min 128, max 512, must be divisible by 32) |

**Docker labels:**
```dockerfile
isAnnotationWorker=""
interfaceName="CondensateNet training"
interfaceCategory="CondensateNet"
description="Fine-tune CondensateNet on user-corrected condensate annotations"
annotationShape="polygon"
```

#### Compute Flow

```
1. Validate inputs
   ├── Check annotation_tag is set → sendError if empty
   └── Check region_tag is set → sendError if empty

2. Collect annotations
   ├── Fetch all polygon annotations from dataset
   ├── Filter by annotation_tag → these are ground-truth condensate boundaries
   ├── Fetch polygon + rectangle annotations
   ├── Filter by region_tag → these are training regions
   └── Validate non-empty results

3. Build training data (for each region)
   ├── Get image at region's (XY, Z, Time, Channel)
   ├── Crop image to region bounding box
   ├── Group training annotations by (Time, XY, Z)
   ├── Get training annotations at same location
   ├── For each annotation in region:
   │   ├── Convert coordinates: {x, y} → [row, col] = [y, x]
   │   ├── Offset to region-local coords (subtract min_y, min_x)
   │   ├── Rasterize polygon → binary mask via skimage.draw.polygon2mask()
   │   └── Assign unique instance ID in label mask
   └── Append (cropped_image, label_mask) to training lists

4. Validate training data
   ├── Check at least one region has annotations
   └── Check total instances > 0

5. Load initial model
   ├── "pretrained" → use baked-in /models/condensatenet
   └── Custom name → download from Girder via utils.download_girder_model()

6. Train
   ├── Create dataset: condensatenet.data.prepare_training_data(images, label_masks, ...)
   ├── Call condensatenet.training.train_model(dataset, initial_model_path, output_path, ...)
   └── Pass progress_callback wrapping sendProgress for UI updates

7. Upload trained model
   └── utils.upload_girder_model(gc, new_model_name, output_path)
```

#### Polygon-to-Mask Rasterization (Step 3 Detail)

This follows the same pattern as `cellpose_train/entrypoint.py` (lines 265-270):

```python
# For each region:
label_mask = np.zeros(cropped_image.shape[:2], dtype=np.int32)
instance_id = 1

for ann in training_annotations_at_this_location:
    # Annotation coordinates are {x, y} in image space
    # polygon2mask expects [row, col] = [y, x]
    polygon_coords = np.array([
        [c['y'] - min_y, c['x'] - min_x]  # offset to region-local coords
        for c in ann['coordinates']
    ])

    # Clip to region bounds (annotations may extend slightly beyond region)
    polygon_coords[:, 0] = np.clip(polygon_coords[:, 0], 0, cropped_image.shape[0] - 1)
    polygon_coords[:, 1] = np.clip(polygon_coords[:, 1], 0, cropped_image.shape[1] - 1)

    mask = draw.polygon2mask(cropped_image.shape[:2], polygon_coords)
    label_mask[mask] = instance_id
    instance_id += 1
```

**Coordinate notes:**
- No 0.5 pixel offset needed for rasterization (unlike property workers that sample pixel intensities). `polygon2mask` treats coordinates as polygon vertices.
- The `{x, y}` → `[y, x]` swap follows the standard convention in this codebase (see CLAUDE.md "Coordinate Conventions" section).
- Annotations are filtered to those inside the region by checking their location matches `(Time, XY, Z)`. Spatially, they should be inside the region polygon, but clipping handles edge cases.

---

### 3. Prediction Worker Updates — `predict/entrypoint.py`

Three targeted changes to the existing `entrypoint.py`:

#### 3a. Add Model Selector to Interface

Add a `'Model'` select field at `displayOrder: 0` (shift all existing fields down by 1):

```python
import utils

def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)

    # List available models: pretrained + any custom models in Girder
    models = ['pretrained']
    girder_models = [m['model_name'] for m in utils.list_girder_models(client.client)[0]]
    models = sorted(list(set(models + girder_models)))

    interface = {
        'Model': {
            'type': 'select',
            'items': models,
            'default': 'pretrained',
            'tooltip': 'Select the model to use. "pretrained" is the default CondensateNet model. '
                       'Custom models from training runs are also available.',
            'noCache': True,
            'displayOrder': 0,
        },
        # ... all existing fields with displayOrder incremented by 1 ...
    }
```

#### 3b. Add Model Download in Compute

```python
def compute(datasetId, apiUrl, token, params):
    worker = WorkerClient(datasetId, apiUrl, token, params)

    model_name = worker.workerInterface.get('Model', 'pretrained')

    # Determine model path
    if model_name == 'pretrained':
        model_path = '/models/condensatenet'  # Baked into Docker image
    else:
        gc = ... # get Girder client
        utils.download_girder_model(gc, model_name)
        model_path = str(utils.MODELS_DIR / model_name)

    # ... rest of compute uses model_path ...
```

#### 3c. Parameterize Model Path in `condensatenet_segmentation()`

Currently hardcodes `/models/condensatenet`:

```python
# BEFORE:
def condensatenet_segmentation(prob_threshold, min_size, max_size):
    pipeline = CondensateNetPipeline.from_local(
        "/models/condensatenet",  # hardcoded
        ...
    )

# AFTER:
def condensatenet_segmentation(model_path, prob_threshold, min_size, max_size):
    pipeline = CondensateNetPipeline.from_local(
        model_path,  # parameterized
        ...
    )
```

And update the call site in `compute()`:
```python
condensatenet = condensatenet_segmentation(model_path, prob_threshold, min_size, max_size)
```

**Everything else stays the same** — tiling, stitching, post-processing, batch mode via WorkerClient.

---

### 4. `docker-compose.yaml`

```yaml
services:
  predict:
    build:
      context: ../../..
      dockerfile: workers/annotations/condensatenet/predict/Dockerfile
    image: annotations/condensatenet_predict:latest
  train:
    build:
      context: ../../..
      dockerfile: workers/annotations/condensatenet/train/Dockerfile
    image: annotations/condensatenet_train:latest
```

**Context note**: The Dockerfiles use `COPY ./annotation_utilities`, `COPY ./worker_client`, etc. relative to the project root. The `context: ../../..` points to `ImageAnalysisProject/` to match this. Reference: `workers/annotations/piscis/docker-compose.yaml`.

---

### 5. Dockerfiles

#### `predict/Dockerfile`

Move the current `Dockerfile` to `predict/Dockerfile` with these additions:
- Add `COPY ./workers/annotations/condensatenet/utils.py /` for Girder model management
- All COPY paths and Docker labels stay the same
- Same NVIDIA CUDA base image, same conda env, same entrypoint pattern

#### `train/Dockerfile`

Nearly identical to predict, with these differences:
- Copies `train/entrypoint.py` instead of `predict/entrypoint.py`
- Docker labels:
  ```dockerfile
  interfaceName="CondensateNet training"
  description="Fine-tune CondensateNet on user-corrected condensate annotations"
  ```

Both Dockerfiles:
- Base: `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
- Install: NimbusImage annotation_client, annotation_utilities, DeepTile, worker_client, condensatenet
- Bake pretrained model: `python -m condensatenet download --output /models/condensatenet`
- Entrypoint: `conda run --no-capture-output -n worker python /entrypoint.py`

---

### 6. Build Script Updates

In `build_machine_learning_workers.sh`, replace the single `docker build` command for condensatenet:

```bash
# BEFORE:
echo "Building CondensateNet worker"
docker build . -f ./workers/annotations/condensatenet/$DOCKERFILE \
    -t annotations/condensatenet:latest $NO_CACHE

# AFTER:
echo "Building CondensateNet workers (predict + train)"
docker compose -f ./workers/annotations/condensatenet/docker-compose.yaml build $NO_CACHE
```

---

### 7. Documentation Updates

#### `CONDENSATENET.md`

Add a "Training" section covering:
- Training workflow (annotate condensates → define regions → run training)
- Interface parameters for the training worker
- How to select a custom model in the prediction worker
- Tips: how many annotations/regions are needed, epoch guidelines

#### `REGISTRY.md`

Update the condensatenet entry to reflect both predict and train capabilities.

---

### 8. Edge Cases and Validation

| Scenario | Handling |
|----------|----------|
| No annotation tag selected | `sendError` + return |
| No region tag selected | `sendError` + return |
| No annotations found with tag | `sendError` + return |
| No regions found with tag | `sendError` + return |
| Region has no training annotations inside it | Skip region (log warning), continue with others |
| No valid training data after all regions | `sendError` + return |
| GPU unavailable | Automatic fallback to CPU (`torch.cuda.is_available()`) |
| Training annotations extend beyond region | Clip polygon coordinates to region bounds |
| Malformed polygon (too few vertices, etc.) | Skip with `try/except`, continue |
| Model name already exists in Girder | Overwrite (delete old, upload new) |
| Patch size not divisible by 32 | Enforce in interface with min/max/step, or round in compute |

---

### 9. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `condensatenet/utils.py` | **CREATE** | Girder model management utilities |
| `condensatenet/docker-compose.yaml` | **CREATE** | Multi-container build config |
| `condensatenet/train/Dockerfile` | **CREATE** | Training container |
| `condensatenet/train/entrypoint.py` | **CREATE** | Training worker logic |
| `condensatenet/predict/Dockerfile` | **CREATE** (moved) | Move current `Dockerfile` here, add utils.py COPY |
| `condensatenet/predict/entrypoint.py` | **CREATE** (moved+modified) | Move current `entrypoint.py`, add model selection |
| `condensatenet/Dockerfile` | **DELETE** | Replaced by `predict/Dockerfile` |
| `condensatenet/entrypoint.py` | **DELETE** | Replaced by `predict/entrypoint.py` |
| `condensatenet/CONDENSATENET.md` | **MODIFY** | Add training documentation |
| `build_machine_learning_workers.sh` | **MODIFY** | Use docker-compose for condensatenet |
| `REGISTRY.md` | **MODIFY** | Update condensatenet entry |

---

### 10. Testing Plan

1. **Build**: `docker compose -f ./workers/annotations/condensatenet/docker-compose.yaml build` — both containers build successfully
2. **Predict (pretrained)**: Run prediction with "pretrained" model selected — should work identically to current behavior
3. **Train**: Create polygon annotations on condensates, define training regions, run training worker — model uploads to Girder
4. **Predict (custom)**: Select the trained model in predict dropdown — inference runs with custom model
5. **Build script**: `./build_machine_learning_workers.sh` builds both containers alongside other ML workers
