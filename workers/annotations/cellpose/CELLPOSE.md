# Cellpose Worker

This worker runs [Cellpose](https://www.cellpose.org/) models to segment cells or nuclei in microscopy images, producing polygon annotations.

## How It Works

1. **Model Selection**: Loads a built-in Cellpose model (cyto, cyto2, cyto3, nuclei) or a user-trained model stored in Girder
2. **Tiling**: Splits the image into overlapping tiles using DeepTile for memory-efficient processing
3. **Segmentation**: Runs Cellpose inference on each tile with GPU acceleration
4. **Stitching**: Merges polygons spanning tile boundaries using DeepTile's `stitch_polygons()`
5. **Post-processing**: Applies optional padding (dilation/erosion) and smoothing (polygon simplification)

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Cellpose** | notes | -- | Informational text with documentation link |
| **Batch XY** | text | -- | XY positions to iterate over (e.g., "1-3, 5-8") |
| **Batch Z** | text | -- | Z slices to iterate over |
| **Batch Time** | text | -- | Time points to iterate over |
| **Model** | select | cyto3 | Cellpose model to use. Includes built-in models and user-trained models from Girder |
| **Primary Channel** | channel | -- | Main segmentation channel. Use cytoplasm channel for cyto models, nucleus channel for nuclei model |
| **Secondary Channel** | channel | -1 (none) | Optional secondary channel. Use nucleus channel when segmenting cytoplasm with cyto models |
| **Diameter** | number | 10 | Expected cell diameter in pixels. Accuracy improves when this is close to actual cell size (range: 0-200) |
| **Smoothing** | number | 0.7 | Polygon simplification tolerance. Higher values produce simpler polygons (range: 0-10) |
| **Padding** | number | 0 | Expand (positive) or shrink (negative) polygons in pixels (range: -20 to 20) |
| **Tile Size** | number | 1024 | Tile dimension in pixels. Reduce if running out of GPU memory (range: 0-2048) |
| **Tile Overlap** | number | 0.1 | Fraction of overlap between adjacent tiles. Objects must be smaller than the overlap region (range: 0-1) |

## Implementation Details

### Model Selection

- **Built-in models**: `cyto`, `cyto2`, `cyto3`, `nuclei` are available without downloading
- **User-trained models**: Listed from the user's Girder `.cellpose/models` folder and downloaded on demand
- The model list is dynamically populated and marked `noCache` so it refreshes each time

### Channel Configuration

- For **cyto/cyto2/cyto3** models: put the cytoplasm channel in Primary and optionally the nucleus channel in Secondary
- For the **nuclei** model: put the nucleus channel in Primary and leave Secondary blank
- If only Primary is provided, Cellpose runs in single-channel mode (`channels=[0,0]`); with both channels, it uses `channels=[1,2]`

### GPU Handling

The worker always requests GPU mode (`gpu=True`). Cellpose handles the fallback to CPU internally if no GPU is available.

### Tiling

Uses DeepTile to split images into square tiles with configurable size and overlap. At 1024px tile size with 0.1 overlap, the overlap region is ~102 pixels. Objects larger than the overlap region may not stitch correctly.

### Polygon Post-processing

Applied in order: padding (via Shapely `buffer()`), then smoothing (via Shapely `simplify()`).

## Notes

- Uses `WorkerClient` for batch processing across XY/Z/Time positions
- Custom models trained via the `cellpose_train` worker appear automatically in the model dropdown
- The `cyto3` model is recommended as the default for general cell segmentation
