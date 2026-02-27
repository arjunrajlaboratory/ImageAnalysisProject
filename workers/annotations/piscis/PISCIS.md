# Piscis Worker

This worker provides two tools -- **Piscis Predict** and **Piscis Train** -- for automated spot/point detection in fluorescence microscopy images using the [Piscis](https://github.com/zjniu/piscis) deep learning model.

## How It Works

### Predict

1. Downloads the selected model from Girder (if not already local).
2. Loads the Piscis model onto GPU (CUDA) if available, otherwise CPU.
3. For each image frame (iterating over Batch XY/Z/Time), runs `model.predict()` to detect point coordinates.
4. Adds a 0.5-pixel offset to coordinates to convert from pixel-center to NimbusImage's coordinate system.
5. Uploads detected points as point annotations via `WorkerClient.process()`.

In **Z-Stack mode**, all Z slices are passed to the model at once for 3D detection. In **Current Z mode**, each Z slice is segmented independently.

### Train

1. Collects point annotations (filtered by Annotation Tag) and region polygons (filtered by Region Tag) from the dataset.
2. For each region, crops the image to the polygon bounding box, applies a rasterized mask, and extracts the points within the region.
3. Coordinates are snapped, fit, and deduplicated using Piscis utilities (`snap_coords`, `fit_coords`, `remove_duplicate_coords`).
4. Generates a training dataset from the collected image/coordinate pairs.
5. Downloads the initial model from Girder, then fine-tunes it using `train_model()` with the specified hyperparameters.
6. Uploads the trained model back to Girder under the user's Private/.piscis/models folder.

## Interface Parameters

### Predict

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Piscis** | notes | -- | Informational text with documentation link. |
| **Batch XY** | text | (none) | XY positions to iterate over (e.g., "1-3, 5-8"). |
| **Batch Z** | text | (none) | Z slices to iterate over (e.g., "1-3, 5-8"). Ignored in Z-Stack mode. |
| **Batch Time** | text | (none) | Time points to iterate over (e.g., "1-3, 5-8"). |
| **Model** | select | `20251212` | Pre-trained or user-trained model to use. Dynamically lists local and Girder-hosted models. |
| **Mode** | select | `Current Z` | `Current Z` segments each Z slice independently; `Z-Stack` segments all Z slices together in 3D. |
| **Scale** | number | `1` | Multiplier on detected object size (range 0-5). |
| **Threshold** | number | `0.5` | Detection confidence threshold (range 0-9). Has minimal effect in practice; switch models for different sensitivity. |
| **Skip Frames Without** | tags | (none) | Skip processing frames that contain no annotations with the specified tag(s). If empty, all frames are processed. |

### Train

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Piscis Train** | notes | -- | Informational text with documentation link. |
| **Initial Model Name** | select | `20251212` | Base model to fine-tune from. Lists local and Girder-hosted models. |
| **New Model Name** | text | (current datetime) | Name for the newly trained model (auto-generated timestamp by default). |
| **Annotation Tag** | tags | (none) | Tag(s) identifying the point annotations to use as training ground truth. Required. |
| **Region Tag** | tags | (none) | Tag(s) identifying polygon/rectangle regions that define training areas. Required. |
| **Learning Rate** | text | `0.1` | Training learning rate. |
| **Weight Decay** | text | `0.0001` | Weight decay regularization. |
| **Epochs** | text | `40` | Number of training epochs. |
| **Random Seed** | text | `42` | Random seed for reproducibility. |

## Implementation Details

- Models are stored on the Girder server in `Private/.piscis/models/` (with legacy fallback to `Public/.piscis/models/`). The predict worker downloads models on demand; the train worker uploads new models after training.
- The predict worker uses `WorkerClient` for batch processing, which handles iteration over XY/Z/Time dimensions automatically.
- Training validates that annotation tags, region tags, annotations, and regions are all non-empty before proceeding, sending user-facing error messages via `sendError()` for each failure case.
- Training uses a warmup fraction of 0.1 (10% of epochs) and splits the random seed into two child seeds -- one for dataset generation and one for model training.
- Both predict and train automatically use CUDA if a GPU is available, falling back to CPU otherwise.

## Notes

- The predict worker produces **point** annotations (not polygons), making it suited for spot detection (e.g., FISH spots, puncta) rather than cell segmentation.
- Training requires at least one point annotation inside every region polygon. Regions without points will cause an error.
- The `Scale` parameter is a multiplier on object size in the model, not a pixel value. The `Threshold` parameter has limited practical effect according to the interface tooltip.
