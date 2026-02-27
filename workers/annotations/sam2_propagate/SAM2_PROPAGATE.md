# SAM2 Propagate

This worker uses SAM2's image predictor to propagate existing annotations through time or Z-slices. It takes annotations at one frame and uses their bounding boxes as prompts to segment the corresponding objects in subsequent (or preceding) frames, creating a chain of tracked annotations.

## How It Works

1. **Annotation Retrieval**: Fetches all polygon annotations matching the specified propagation tag
2. **Optional Resegmentation**: If enabled, re-runs SAM2 on the source annotations at their current frame to get cleaner masks before propagation
3. **Frame-by-Frame Propagation**: For each frame in the batch sequence:
   - Finds annotations at the current frame (either original or newly created)
   - Converts annotation polygons to bounding boxes
   - Loads the next frame's image (merged from all channels via layer settings)
   - Runs `SAM2ImagePredictor.predict()` with the bounding boxes as prompts
   - Converts output masks to polygon annotations at the next frame's location
4. **Connection Creation**: If "Connect sequentially" is enabled, creates parent-child connections between annotations across frames using temporary IDs

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **SAM2 Propagate** | notes | -- | Description of the tool's purpose |
| **Batch XY** | text | Current tile | XY positions to process (e.g., "1-3, 5-8") |
| **Batch Z** | text | Current tile | Z positions to process/propagate through |
| **Batch Time** | text | Current tile | Time positions to process/propagate through |
| **Propagate across** | select | Time | Dimension to propagate along: "Time" or "Z" |
| **Propagation direction** | select | Forward | Direction of propagation: "Forward" or "Backward" |
| **Tag of objects to propagate** | tags | -- | Tag(s) identifying source annotations to propagate |
| **Model** | select | `sam2.1_hiera_small.pt` | SAM2.1 model checkpoint |
| **Resegment propagation objects** | checkbox | True | Re-run SAM2 on source annotations before propagating |
| **Connect sequentially** | checkbox | True | Create parent-child connections between frames |
| **Padding** | number | 0 | Expand (+) or shrink (-) polygons in pixels (-20 to 20) |
| **Smoothing** | number | 0.7 | Polygon simplification tolerance (0 to 3) |

## Implementation Details

### Propagation Mechanism

Uses SAM2's `SAM2ImagePredictor` (not the video predictor). At each step, it takes bounding boxes from the previous frame's annotations and uses them as box prompts on the next frame. This means each frame is segmented independently using the prior frame's results as guidance.

When "Backward" direction is selected, the batch sequence along the propagation axis is reversed so that propagation proceeds from later frames to earlier ones.

### Resegmentation

When "Resegment propagation objects" is enabled (default), the worker first re-segments the source annotations at their original location using SAM2 before propagating. This ensures propagation starts from SAM2-consistent masks rather than potentially rough manual annotations. The resegmented annotations are uploaded along with the propagated ones.

### Connection Tracking

The worker uses a temporary ID system (`uuid4`) to track parent-child relationships across frames before upload. After uploading annotations to the server, it maps temporary IDs to server-assigned IDs and creates connection objects with the tag `SAM2_PROPAGATED`. When propagating backward, parent-child direction is reversed so that the earlier frame is always the parent.

### Model Selection

Same SAM2.1 Hiera variants as the automatic mask generator: tiny, small (default), base_plus, and large. Larger models may give better results but use more memory, which can be an issue when propagating many objects simultaneously.

### GPU Handling

Requires CUDA GPU. Enables bfloat16 autocast and TF32 on Ampere GPUs.

## Notes

- This worker uses the SAM2 **image predictor** (frame-by-frame). For the SAM2 **video predictor** which processes all frames jointly with temporal context, see `sam2_video`.
- Useful for time-lapse microscopy where objects change character over time (e.g., cell differentiation, condensate dynamics).
- The source annotations are identified by the "Tag of objects to propagate" field, which is separate from the output tags applied to newly created annotations.
- If no annotations are found at a given frame, that frame is skipped and propagation continues from the next frame that has annotations.
