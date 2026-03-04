# Registration Worker

Registers time-lapse images to correct for drift and movement over time using the pystackreg library. Supports algorithmic registration, manual control points, or a combination of both. Uploads the registered image to Girder.

## How It Works

1. Computes cumulative registration matrices across time points for each XY position, using the specified algorithm on consecutive frames from the reference channel and Z plane
2. Optionally incorporates user-placed control point annotations to guide or override the algorithmic registration
3. If a reference time other than the first is specified, re-references all matrices so that time point is the identity transform
4. Applies the computed transformations to all selected channels and writes the registered result to Girder

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Apply to XY coordinates** | text | all | XY positions to register (e.g., "1-3, 5-7"). Default is all positions. |
| **Reference Z Coordinate** | text | 1 (first Z) | Z plane used for computing registration (1-indexed). |
| **Reference Time Coordinate** | text | 1 (first time) | Time point that remains unchanged; all other frames align to this (1-indexed). |
| **Reference Channel** | channel | -- | Channel used for computing the registration transforms. |
| **Channels to correct** | channelCheckboxes | -- | Which channels to apply the registration transforms to. |
| **Reference region tag** | tags | -- | Tag of a polygon/rectangle annotation defining the region used for registration. If set, only this subregion is used to compute transforms. |
| **Control point tag** | tags | -- | Tag of point annotations used as manual control points for registration. |
| **Apply algorithm after control points** | checkbox | false | If true, applies the selected algorithm after first correcting with control points. |
| **Algorithm** | select | Translation | Registration algorithm: "None (control points only)", "Translation", "Rigid" (translation + rotation), or "Affine" (translation + rotation + scaling). |

## Implementation Details

### Registration Matrix Computation

- Registration matrices are computed cumulatively: each frame is registered relative to the previous frame, and the transforms are multiplied together. This handles gradual drift without requiring every frame to be similar to the reference.
- When a reference time other than t=0 is specified, all matrices are multiplied by the inverse of the matrix at the reference time, making that time point the identity.

### Control Points

- Control points are point annotations with the specified tag. The worker looks up one control point per (XY, Time) combination.
- When control points exist at both t-1 and t, a translation matrix is computed from the coordinate difference.
- If "Apply algorithm after control points" is enabled, the algorithmic registration is run on the control-point-corrected image and the two transforms are composed.
- If control points are missing for a given pair of frames, the worker falls back to the algorithmic registration.

### Reference Region

- If a reference region tag is specified, only the bounding box of the first matching polygon or rectangle annotation is used to compute registration transforms. The full image is still transformed during the apply step.
- Useful when the relevant features for alignment are concentrated in one area of the image.

### Safe Type Casting

- Uses a `safe_astype` function that clips values to the dtype range before casting, preventing integer overflow artifacts in the registered output.

## Notes

- Requires a time dimension; sends an error if the image has no IndexT.
- The registration transform is computed on the reference channel only but applied to all selected channels.
- Unselected channels and XY positions outside the specified range pass through unchanged.
- Output metadata includes the tool name, algorithm, reference coordinates, and which XY positions were registered.
