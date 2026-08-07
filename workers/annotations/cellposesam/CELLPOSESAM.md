# Cellpose-SAM Worker

This worker runs Cellpose-SAM, a variant of Cellpose that combines Cellpose with SAM (Segment Anything Model) for cell segmentation. It supports up to three input channel slots and produces polygon annotations.

## How It Works

1. **Channel Assembly**: Collects up to three input channels from user-selected channel checkboxes and stacks them
2. **Model Selection**: Loads a built-in Cellpose-SAM checkpoint (`cpsam_v2` by default, or the original `cpsam`) or a user-trained model from Girder
3. **Tiling**: Splits the image into overlapping tiles using DeepTile
4. **Segmentation**: Runs Cellpose-SAM inference on each tile with GPU acceleration
5. **Stitching**: Merges polygons spanning tile boundaries using DeepTile's `stitch_polygons()`
6. **Post-processing**: Applies optional padding (dilation/erosion) and smoothing (polygon simplification)

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Cellpose-SAM** | notes | -- | Informational text with documentation link |
| **Batch XY** | text | -- | 1-indexed XY positions to iterate over (e.g., "1-3, 5-8") |
| **Batch Z** | text | -- | 1-indexed Z slices to iterate over |
| **Batch Time** | text | -- | 1-indexed Time points to iterate over |
| **Model** | select | cellpose-sam | Model to use. `cellpose-sam` runs the `cpsam_v2` checkpoint (current default); `cellpose-sam (legacy cpsam)` runs the original April 2025 `cpsam` checkpoint. User-trained models from Girder are also listed |
| **Channel for Slot 1** | channelCheckboxes | -- | **Required.** Source channel(s) for the model's first input slot. If multiple selected, only the first is used |
| **Channel for Slot 2** | channelCheckboxes | -- | Optional second input slot channel |
| **Channel for Slot 3** | channelCheckboxes | -- | Optional third input slot channel |
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
- Each slot value is parsed by `annotation_tools.get_selected_channels()` rather than `.items()`, which crashed when a saved tool config held a bare list (`[0]`) instead of the documented `{"0": True}` mapping. A value in any other shape is reported as an error rather than defaulting to some channel

### Model Behavior

- **Base models**: The dropdown labels map to cellpose built-in checkpoints in `models_config.py` — `cellpose-sam` → `cpsam_v2`, `cellpose-sam (legacy cpsam)` → `cpsam`. The selected checkpoint name is passed explicitly as `pretrained_model` rather than relying on Cellpose's internal default, which can shift between versions.
- **Custom models**: Loaded from Girder by path. Like built-in Cellpose-SAM checkpoints, they run at native resolution with no diameter rescaling.

### Batch Validation

Batch XY/Z/Time values are validated against the dataset before the GPU model
is loaded. The fields use 1-based positions, so `0` is invalid; out-of-range
values produce a user-facing error that includes the dataset's valid range.
Selected input channels are range-checked at the same preflight stage.

### Built-in Checkpoints

`cpsam_v2` (SAM-ViTL backbone, released June 2026) is the default; it reduces spurious masks in low-contrast regions compared to the original `cpsam` (April 2025). Both checkpoints (~1.23 GB each) are pre-downloaded at build time by `download_models.py` so neither downloads on first run. Requires `cellpose==4.2.1.1` (cpsam_v2 was added in the 4.2.x line). To add or change the offered checkpoints, edit `models_config.py` — it is the single source of truth for both the interface and the build-time download.

### GPU Handling

The worker always requests GPU mode (`gpu=True`). Cellpose handles the fallback to CPU internally if no GPU is available.

### Tiling

Uses DeepTile to split images into square tiles with configurable size and overlap. At 1024px tile size with 0.1 overlap, the overlap region is ~102 pixels. Objects larger than the overlap region may not stitch correctly.

### Polygon Post-processing

Applied in order: padding (via Shapely `buffer()`), then smoothing (via Shapely `simplify()`).

### Model selection validation

`compute()` validates the saved `Model` selection with
`annotation_tools.get_required_select()` before loading the model. A saved tool
config can hold `null` for a `select` field even though the interface defines a
default. Previously a `null` model was passed straight to the Girder model
downloader, failing with a confusing error. Invalid selections now fail fast
with `sendError` and re-raise, so the job is recorded as failed rather than
silently succeeding; the fix is to re-select the model in the tool settings and
save the tool again. Because models can be user-trained and downloaded from
Girder, there is no static option list — validation checks only that the
selection is a non-empty string.

## Notes

- Uses `WorkerClient` for batch processing across XY/Z/Time positions
- Custom models trained via the `cellpose_train` worker appear automatically in the model dropdown
- The key difference from the standard Cellpose worker is the multi-slot channel input and the SAM-enhanced base model
