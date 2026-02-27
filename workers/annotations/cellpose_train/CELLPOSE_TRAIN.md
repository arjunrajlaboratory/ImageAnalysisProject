# Cellpose Train Worker

This worker fine-tunes a Cellpose model on user-corrected annotations, producing a custom model that can be used by the Cellpose and Cellpose-SAM workers.

## How It Works

1. **Annotation Loading**: Retrieves all polygon and rectangle annotations from the dataset
2. **Tag Filtering**: Selects training annotations by the specified training tag, and optionally crops to training region annotations
3. **Image Assembly**: For each location (XY/Z/Time), loads primary and optional secondary channel images and renders training annotations into label masks
4. **Region Cropping**: If training regions are specified, crops each image/mask pair to the bounding box of each region polygon
5. **Training**: Fine-tunes the selected base Cellpose model using `cellpose.train.train_seg()` with SGD optimization
6. **Upload**: Saves the trained model to the user's Girder `.cellpose/models` folder

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Cellpose train** | notes | -- | Informational text with documentation link |
| **Base Model** | select | cyto3 | Base model to fine-tune. Includes built-in models (cyto, cyto2, cyto3, nuclei) and user-trained models from Girder |
| **Nuclear Model?** | checkbox | false | Check if training a nuclear segmentation model. Affects channel ordering |
| **Output Model Name** | text | -- | Name for the saved model. Will appear in the model dropdown of Cellpose/Cellpose-SAM workers |
| **Primary Channel** | channel | -- | **Required.** Main channel for training. Cytoplasm channel for cyto models, nucleus channel for nuclear models |
| **Secondary Channel** | channel | -1 (none) | Optional secondary channel. Nucleus channel when training cyto models; ignored for nuclear models |
| **Training Tag** | tags | -- | **Required.** Tag identifying the corrected annotations to use as training ground truth |
| **Training Region** | tags | -- | Optional tag identifying region annotations that define training crops. If empty, uses the full image |
| **Learning Rate** | number | 0.01 | SGD learning rate (range: 0.0001-0.5) |
| **Epochs** | number | 1000 | Number of training epochs (range: 100-2000) |
| **Weight Decay** | number | 0.0001 | Weight decay regularization (range: 0-0.01) |

## Implementation Details

### Training Data Preparation

- Annotations are grouped by location (Time, Z, XY) to batch image loading
- Both polygon and rectangle annotations are retrieved and combined
- Each training annotation is rasterized into a label mask using `skimage.draw.polygon2mask()`, with each annotation assigned a unique integer label
- Images are assembled as 3-channel stacks: `[primary, secondary, zeros]`

### Training Regions

- Training regions are polygon/rectangle annotations with the specified region tag
- When regions are specified, images and label masks are cropped to each region's bounding box, producing one training sample per region per location
- When no regions are specified, the full image is used as a single training sample (a warning is displayed)
- Using regions is recommended to focus training on relevant areas and reduce memory usage

### Channel Configuration

- **Nuclear model** (`Nuclear Model?` checked): Uses `channels=[1, 0]` (primary channel only)
- **Cytoplasm model** with secondary channel: Uses `channels=[1, 2]` (primary + secondary)
- **Cytoplasm model** without secondary channel: Falls back to `channels=[1, 0]` with a warning

### Model Storage

- Trained models are saved locally to `/root/.cellpose/models/` during training
- After training completes, the model is uploaded to the user's Girder `Private/.cellpose/models/` folder
- If a model with the same name already exists in Girder, it is replaced
- Uploaded models automatically appear in the model dropdowns of the Cellpose and Cellpose-SAM workers

### GPU Handling

The worker checks for GPU availability via `cellpose.core.use_gpu()` and logs the result, but model instantiation uses the default Cellpose behavior (GPU if available, CPU otherwise).

## Notes

- Does not use `WorkerClient` for batch processing; instead directly manages annotation retrieval and image loading
- A training tag is required; the worker will error if none is provided
- Training regions are optional but recommended for efficiency and to avoid training on irrelevant parts of the image
- The base model list is dynamically populated from both built-in models and user models in Girder
- Progress updates are sent at key stages: loading annotations (10%), processing (20%), loading images (30%), training (40%), and saving (95%)
