# Cellpose-SAM Worker

This worker runs Cellpose-SAM, a variant of Cellpose that combines Cellpose with SAM (Segment Anything Model) for cell segmentation. It supports up to three input channel slots and produces polygon annotations.

## How It Works

1. **Channel Assembly**: Collects up to three input channels from user-selected channel checkboxes and stacks them
2. **Model Selection**: Loads the built-in `cellpose-sam` model or a user-trained model from Girder
3. **Tiling**: Splits the image into overlapping tiles using DeepTile
4. **Segmentation**: Runs Cellpose-SAM inference on each tile with GPU acceleration
5. **Stitching**: Merges polygons spanning tile boundaries using DeepTile's `stitch_polygons()`
6. **Post-processing**: Applies optional padding (dilation/erosion) and smoothing (polygon simplification)

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Cellpose-SAM** | notes | -- | Informational text with documentation link |
| **Batch XY** | text | -- | XY positions to iterate over (e.g., "1-3, 5-8") |
| **Batch Z** | text | -- | Z slices to iterate over |
| **Batch Time** | text | -- | Time points to iterate over |
| **Model** | select | cellpose-sam | Model to use. Includes built-in `cellpose-sam` and user-trained models from Girder |
| **Channel for Slot 1** | channelCheckboxes | -- | **Required.** Source channel(s) for the model's first input slot. If multiple selected, only the first is used |
| **Channel for Slot 2** | channelCheckboxes | -- | Optional second input slot channel |
| **Channel for Slot 3** | channelCheckboxes | -- | Optional third input slot channel |
| **Diameter** | number | 10 | Expected cell diameter in pixels (range: 0-200). Only used for custom models, not the base `cellpose-sam` model |
| **Smoothing** | number | 0.7 | Polygon simplification tolerance (range: 0-10) |
| **Padding** | number | 0 | Expand (positive) or shrink (negative) polygons in pixels (range: -20 to 20) |
| **Tile Size** | number | 1024 | Tile dimension in pixels (range: 0-2048) |
| **Tile Overlap** | number | 0.1 | Fraction of overlap between adjacent tiles (range: 0-1) |

## Implementation Details

### Channel Slots vs. Primary/Secondary Channels

Unlike the standard Cellpose worker which uses Primary/Secondary channel selectors, Cellpose-SAM uses three `channelCheckboxes` slots. This allows flexible multi-channel input:

- **Slot 1** is required; an error is raised if no channel is selected
- **Slots 2 and 3** are optional
- If multiple channels are checked in a single slot, a warning is issued and only the first is used
- Selected channels are stacked in order and passed to the model

### Model Behavior

- **Base model** (`cellpose-sam`): Runs with `gpu=True` and no diameter or channel parameters in `eval_parameters`, relying on the model's built-in defaults
- **Custom models**: Loaded from Girder by path and use the user-specified diameter in `eval_parameters`

### GPU Handling

The worker always requests GPU mode (`gpu=True`). Cellpose handles the fallback to CPU internally if no GPU is available.

### Tiling

Uses DeepTile to split images into square tiles with configurable size and overlap. At 1024px tile size with 0.1 overlap, the overlap region is ~102 pixels. Objects larger than the overlap region may not stitch correctly.

### Polygon Post-processing

Applied in order: padding (via Shapely `buffer()`), then smoothing (via Shapely `simplify()`).

## Notes

- Uses `WorkerClient` for batch processing across XY/Z/Time positions
- Custom models trained via the `cellpose_train` worker appear automatically in the model dropdown
- The key difference from the standard Cellpose worker is the multi-slot channel input and the SAM-enhanced base model
