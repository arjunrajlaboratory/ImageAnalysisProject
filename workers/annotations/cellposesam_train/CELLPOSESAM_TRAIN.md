# Cellpose-SAM Retrain Worker

This worker fine-tunes a Cellpose-SAM model on user-corrected annotations, producing a custom model that can be used by the Cellpose-SAM worker.

It is the Cellpose-SAM counterpart to the older [Cellpose Train worker](../cellpose_train/CELLPOSE_TRAIN.md). It follows the same annotation-gathering paradigm (training tag + optional training regions) but adopts the Cellpose-SAM interface and training API, which differ substantially from earlier Cellpose versions.

## How It Works

1. **Annotation Loading**: Retrieves all polygon and rectangle annotations from the dataset
2. **Tag Filtering**: Selects training annotations by the specified training tag, and optionally crops to training region annotations
3. **Image Assembly**: For each location (XY/Z/Time), loads the selected input-slot channels and stacks them channels-last, then renders training annotations into label masks
4. **Region Cropping**: If training regions are specified, crops each image/mask pair to the bounding box of each region polygon
5. **Fine-tuning**: Fine-tunes the selected base Cellpose-SAM model using `cellpose.train.train_seg()` (AdamW optimizer)
6. **Upload**: Saves the trained model to the user's Girder `.cellposesam/models` folder

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Cellpose-SAM retrain** | notes | -- | Informational text with documentation link |
| **Base Model** | select | cellpose-sam | Base model to fine-tune. `cellpose-sam` starts from the `cpsam_v2` checkpoint; `cellpose-sam (legacy cpsam)` starts from the original April 2025 `cpsam` checkpoint. User-trained models from Girder are also listed |
| **Output Model Name** | text | -- | **Required.** Plain filename for the saved model, without path separators. Will appear in the Model dropdown of the Cellpose-SAM worker. Built-in model labels are reserved and cannot be used as custom output names |
| **Channel for Slot 1** | channelCheckboxes | -- | **Required.** Source channel(s) for the model's first input slot. If multiple selected, only the first is used |
| **Channel for Slot 2** | channelCheckboxes | -- | Optional second input slot channel |
| **Channel for Slot 3** | channelCheckboxes | -- | Optional third input slot channel |
| **Training Tag** | tags | -- | **Required.** Tag identifying the corrected annotations to use as training ground truth |
| **Training Region** | tags | -- | Optional tag identifying region annotations that define training crops. If empty, uses the full image |
| **Learning Rate** | number | 0.00001 | AdamW learning rate (range: 0.000001-0.1). Cellpose-SAM fine-tunes best with a small learning rate (1e-5) |
| **Epochs** | number | 100 | Number of training epochs (range: 10-2000) |
| **Weight Decay** | number | 0.1 | AdamW weight decay regularization (range: 0-1) |

## Implementation Details

### Channel Slots vs. Primary/Secondary Channels

Unlike the older `cellpose_train` worker (which used Primary/Secondary channel selectors and a "Nuclear Model?" checkbox to build the classic `channels=[cyto, nucleus]` pair), Cellpose-SAM works directly on the channels you provide. It ingests up to three channels natively and reorders/normalizes them internally, so there is no separate cytoplasm/nucleus concept.

This worker therefore uses the same three `channelCheckboxes` slots as the Cellpose-SAM inference worker:

- **Slot 1** is required; an error is raised if no channel is selected
- **Slots 2 and 3** are optional
- If multiple channels are checked in a single slot, a warning is issued and only the first is used
- Selected channels are stacked channels-last and passed to training via `channel_axis=-1`
- Each slot value is parsed by `annotation_tools.get_selected_channels()` rather than `.items()`, which crashed when a saved tool config held a bare list (`[0]`) instead of the documented `{"0": True}` mapping. A value in any other shape is reported as an error rather than defaulting to some channel

Matching the inference worker's channel layout matters: a model trained on a given channel arrangement should be run with the same arrangement.

### Training Data Preparation

- Annotations are grouped by location (Time, Z, XY) to batch image loading
- Both polygon and rectangle annotations are retrieved and combined
- Each training annotation is rasterized into a label mask using `skimage.draw.polygon2mask()`, with each annotation assigned a unique integer label
- Images are stacked channels-last as `(H, W, num_selected_channels)`; Cellpose-SAM standardizes to three channels internally (padding with zeros or truncating as needed)

### Training Regions

- Training regions are polygon/rectangle annotations with the specified region tag
- When regions are specified, images and label masks are cropped to each region's bounding box, producing one training sample per region per location
- When no regions are specified, the full image is used as a single training sample (a warning is displayed)
- Using regions is recommended to focus training on relevant areas and reduce memory usage

### Cellpose-SAM Training API Differences

Cellpose-SAM (cellpose ≥ 4) changed the training API relative to the versions used by `cellpose_train`:

- **No `channels` argument**: the network ingests channels directly, so training passes `channel_axis=-1` instead of a `channels=[...]` pair
- **AdamW optimizer**: the old `SGD` flag is deprecated (AdamW is always used), so it is not passed
- **Small learning rate**: the recommended default is `1e-5` (vs `0.01` for the older worker)
- **`weight_decay=0.1`** is the Cellpose-SAM default (vs `0.0001`)
- **`bsize=256`**: cpsam requires a block size of 256
- **Native resolution**: training runs with `rescale=False`, so no diameter parameter is needed
- **Model instantiation**: `CellposeModel(gpu=..., pretrained_model=<checkpoint-or-path>)` replaces the old `model_type=` argument

### Model Storage

- The trained model is saved locally to `/root/.cellposesam/models/` (via `save_path` + `model_name` passed to `train_seg`)
- After training completes, the model is uploaded to the user's Girder `Private/.cellposesam/models/` folder
- If a model with the same name already exists in Girder, it is replaced
- Uploaded models automatically appear in the Model dropdown of the Cellpose-SAM worker

### Built-in Checkpoints

The base-model labels map to cellpose built-in checkpoints in `models_config.py` — `cellpose-sam` → `cpsam_v2`, `cellpose-sam (legacy cpsam)` → `cpsam`. Both checkpoints are pre-downloaded at build time by `download_models.py` so neither downloads on first run. `models_config.py` is kept in sync with the inference worker and is the single source of truth for both the interface and the build-time download.

### GPU Handling

The worker checks for GPU availability via `cellpose.core.use_gpu()` and instantiates the model with that result; cellpose falls back to CPU if no GPU is available.

### Model selection validation

`compute()` validates the saved `Base Model` selection with
`annotation_tools.get_required_select()` before loading the model. A saved tool
config can hold `null` for a `select` field even though the interface defines a
default. Previously a `null` model was passed straight to the Girder model
downloader, failing with a confusing error. Invalid selections now fail fast
with `sendError` and re-raise, so the job is recorded as failed rather than
silently succeeding; the fix is to re-select the model in the tool settings and
save the tool again. Because models can be user-trained and downloaded from
Girder, there is no static option list — validation checks only that the
selection is a non-empty string.

## Testing

`tests/test_models_config.py` unit-tests the base-model mapping (`models_config.py`) in isolation — it has no heavy imports, so it runs in the lightweight local venv:

```bash
.cache/testvenv/bin/pytest workers/annotations/cellposesam_train/tests -q
```

## Notes

- Does not use `WorkerClient` for batch processing; instead directly manages annotation retrieval and image loading (same approach as `cellpose_train`)
- Requires a training tag and an output model name; the worker errors if either is missing
- Training regions are optional but recommended for efficiency and to avoid training on irrelevant parts of the image
- The base-model list is dynamically populated from both built-in checkpoints and user models in Girder
- Progress updates are sent at key stages: loading annotations (10%), processing (20%), loading images (30%), training (40%), and saving (95%)
- Related workers: [Cellpose-SAM](../cellposesam/CELLPOSESAM.md) (inference), [Cellpose Train](../cellpose_train/CELLPOSE_TRAIN.md) (older retrain)
