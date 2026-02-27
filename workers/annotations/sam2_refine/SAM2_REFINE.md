# SAM2 Refine

This worker uses SAM2's image predictor to refine existing polygon annotations. It takes each annotation's bounding box as a prompt and generates a new, cleaner segmentation mask that better matches the underlying image data. Optionally deletes the original annotations after refinement.

## How It Works

1. **Annotation Retrieval**: Fetches all polygon (and blob) annotations matching the specified refinement tag, filtered to the batch range
2. **Grouping**: Groups annotations by their (XY, Z, Time) location to minimize redundant image loading
3. **Per-Location Processing**: For each location:
   - Loads the merged multi-channel image via layer settings
   - Converts each annotation to a bounding box
   - Runs `SAM2ImagePredictor.predict()` individually per bounding box with `multimask_output=False`
   - Converts output masks to simplified polygons with optional padding
4. **Upload**: Creates new refined annotations and optionally deletes the originals

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **SAM2 Refiner** | notes | -- | Description of the tool's purpose |
| **Batch XY** | text | Current tile | XY positions to process (e.g., "1-3, 5-8") |
| **Batch Z** | text | Current tile | Z positions to process |
| **Batch Time** | text | Current tile | Time positions to process |
| **Tag of objects to refine** | tags | -- | Tag(s) identifying annotations to refine |
| **Model** | select | `sam2.1_hiera_small.pt` | SAM2.1 model checkpoint |
| **Delete original annotations** | checkbox | False | Delete originals after creating refined versions |
| **Padding** | number | 0 | Expand (+) or shrink (-) refined polygons in pixels (-20 to 20) |
| **Smoothing** | number | 0.7 | Polygon simplification tolerance (0 to 3) |

## Implementation Details

### Refinement Strategy

Each annotation is processed individually: its bounding box is extracted and used as a single box prompt to SAM2. The model returns a single mask (`multimask_output=False`) which is then converted to a polygon. This approach works well for cleaning up rough manual annotations or annotations from other segmentation tools.

### Blob Annotation Support

In addition to polygon annotations, the worker also attempts to fetch and refine blob-shaped annotations. If the server does not support the blob shape, the error is silently caught and only polygons are processed.

### GPU Handling

Detects CUDA availability at runtime and falls back to CPU if no GPU is available. When CUDA is present, enables bfloat16 autocast and TF32 on Ampere GPUs. This is the only SAM2 worker with explicit CPU fallback.

### Error Handling

The worker includes per-annotation error handling: if SAM2 fails to process a particular annotation (e.g., degenerate bounding box), it prints a warning and continues with the remaining annotations rather than failing the entire job.

### Annotation Grouping

Annotations are grouped by (XY, Z, Time) location before processing. The image for each location is loaded only once and the SAM2 image embedding is computed once per location, then reused for all annotations at that location. This is significantly more efficient than loading the image per annotation.

### Model Selection

Same SAM2.1 Hiera variants as other SAM2 workers: tiny, small (default), base_plus, and large. Checkpoints are auto-detected from `/code/sam2/checkpoints/`.

## Notes

- The refined annotations are created with the output tags from the tool configuration, not necessarily the same tags as the originals. Make sure to set appropriate output tags.
- When "Delete original annotations" is enabled, originals are deleted only after all refined annotations have been successfully uploaded.
- This worker creates entirely new annotations rather than modifying existing ones in place. Connections or other references to the original annotations will not automatically transfer to the refined versions.
- Related workers: `sam2_propagate` (propagate annotations across frames), `sam2_automatic_mask_generator` (segment without existing annotations).
