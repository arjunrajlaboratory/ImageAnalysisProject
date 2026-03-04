# SAM2 Automatic Mask Generator

This worker uses Meta's [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2) to automatically detect and segment all objects in an image. It is the SAM2 successor to the SAM1 `sam_automatic_mask_generator` worker, with batch processing support and configurable grid density.

## How It Works

1. **Image Loading**: For each batch position, loads all channels and merges them using the dataset's layer settings (contrast, color) via `annotation_tools.process_and_merge_channels`
2. **SAM2 Inference**: Runs `SAM2AutomaticMaskGenerator` with a configurable points-per-side grid to generate masks for all detected objects
3. **Polygon Extraction**: Converts each binary mask to contours, takes the first (largest) contour per mask, simplifies with Shapely, and uploads as polygon annotations
4. **Batch Processing**: Iterates over all XY/Z/Time combinations specified in the batch parameters

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Batch XY** | text | Current tile | XY positions to process (e.g., "1-3, 5-8") |
| **Batch Z** | text | Current tile | Z positions to process |
| **Batch Time** | text | Current tile | Time positions to process |
| **Model** | select | `sam2.1_hiera_small.pt` | SAM2.1 model checkpoint to use |
| **Smoothing** | number | 0.3 | Polygon simplification tolerance (0 to 3) |
| **Points per side** | number | 32 | Grid density for automatic mask generation (16 to 128) |

## Implementation Details

### Model Selection

Available SAM2.1 Hiera model variants (auto-detected from `/code/sam2/checkpoints/`):

| Checkpoint | Config | Size |
|------------|--------|------|
| `sam2.1_hiera_tiny.pt` | `sam2.1_hiera_t.yaml` | Smallest/fastest |
| `sam2.1_hiera_small.pt` | `sam2.1_hiera_s.yaml` | Default, good balance |
| `sam2.1_hiera_base_plus.pt` | `sam2.1_hiera_b+.yaml` | Larger |
| `sam2.1_hiera_large.pt` | `sam2.1_hiera_l.yaml` | Best quality, most memory |

### GPU Handling

Requires CUDA GPU. Enables bfloat16 autocast for performance and enables TF32 on Ampere (compute capability 8+) GPUs.

### Image Compositing

Unlike the SAM1 worker which manually combines two channels, this worker uses the NimbusImage layer system to merge all channels according to user-configured contrast and color settings, producing a float32 RGB image suitable for SAM2 input.

### Points Per Side

Controls the density of the point grid used to prompt the automatic mask generator. Higher values detect more/smaller objects but increase computation time. The default of 32 means a 32x32 grid (1024 points) is used to seed mask generation.

## Notes

- For best results, adjust the layer contrast settings in NimbusImage before running this worker, since the merged channel image is what SAM2 sees.
- Only the first (largest) contour per mask is converted to a polygon; internal holes or secondary contours are discarded.
- Related workers: `sam_automatic_mask_generator` (SAM1 version), `sam2_fewshot_segmentation` (prompt-based with training examples).
