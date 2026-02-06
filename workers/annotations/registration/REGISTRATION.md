# Registration Worker

This worker aligns time-lapse images by computing registration matrices between consecutive timepoints and applying the resulting transformations to produce a corrected image stack. It uses [pystackreg](https://github.com/glichtner/pystackreg) for matrix computation and `scipy.ndimage.affine_transform` for memory-efficient image transformation.

## How It Works

The worker operates in three phases:

1. **Matrix computation**: For each XY position, iterates through timepoints and computes a cumulative 3x3 registration matrix relative to t=0 (or a user-specified reference time). Uses `pystackreg` to register consecutive frames on a single reference channel. Optionally uses control points for manual alignment and/or a reference region to focus registration on a subregion of the image.
2. **Reference time adjustment**: If a reference time other than t=0 is specified, all matrices are multiplied by the inverse of the reference time's matrix so that the reference frame remains unchanged.
3. **Output**: Iterates through every frame in the dataset. For frames in selected channels and XY positions, applies the registration matrix via `apply_transform()`. Writes the result to a TIFF file and uploads it back to Girder.

## Interface Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| **Apply to XY coordinates** | text | XY positions to register (e.g., "1-3, 5-7") | All |
| **Reference Z Coordinate** | text | Z plane used for computing registration | 1 |
| **Reference Time Coordinate** | text | Timepoint treated as the fixed reference | 1 |
| **Reference Channel** | channel | Channel used for computing registration matrices | First channel |
| **Channels to correct** | channelCheckboxes | Which channels to apply the transformation to | None (required) |
| **Reference region tag** | tags | Tag of a polygon/rectangle annotation defining the subregion to use for registration | None |
| **Control point tag** | tags | Tag of point annotations for manual alignment | None |
| **Apply algorithm after control points** | checkbox | Run algorithmic registration on top of control point alignment | False |
| **Algorithm** | select | Registration constraint: None (control points only), Translation, Rigid, Affine | Translation |

## Implementation Details

### Registration Algorithms

The Algorithm selector maps to `pystackreg.StackReg` modes:

- **Translation**: Corrects x/y shifts only (2 DOF)
- **Rigid**: Corrects shifts and rotation (3 DOF)
- **Affine**: Corrects shifts, rotation, and scaling/shear (6 DOF)
- **None (control points only)**: Skips algorithmic registration; uses only manually placed control points

### Control Points

When control point tags are provided, the worker looks up point annotations at each (XY, Time) location and computes a translation matrix from the displacement between consecutive timepoints. If "Apply algorithm after control points" is checked, the algorithmic registration runs on the control-point-corrected image, and the two matrices are composed.

Note: `sr.transform()` (pystackreg) is still used in the matrix computation phase when applying the algorithm after control points, since it operates on the (possibly cropped) reference region and memory is not an issue there.

### Auto-Crop for Large Images

When no reference region is specified and the image exceeds 2048 pixels in either dimension, the worker automatically crops to a center 2048x2048 region for the matrix computation phase only. This prevents out-of-memory crashes during `pystackreg`'s `register()` and `transform()` calls, which internally convert images to float64. A warning is sent to the user explaining the auto-crop and suggesting they specify a Reference region tag to control which region is used.

The full-resolution image is still used in the output phase (phase 3), where the memory-efficient `apply_transform()` function handles the transformation.

### Memory-Efficient Output Transform

The `apply_transform()` helper replaces `pystackreg`'s `sr.transform()` in the output loop. Key differences:

- Uses **float32** instead of pystackreg's float64, cutting peak memory per frame roughly in half
- Uses `scipy.ndimage.affine_transform` with bilinear interpolation (`order=1`)
- Explicitly deletes intermediate arrays and calls `gc.collect()` after each frame

This was implemented to fix an OOM crash (Docker exit code 137) when processing a 12089x12089 image with 7 channels and 2 timepoints (14 total frames). At float64, a single frame required ~2.9 GB of temporary arrays, exceeding the container's memory limit.

### Output File

The worker writes a registered TIFF to `/tmp/registered.tiff`, uploads it to the same Girder folder as the source dataset, and attaches metadata recording the tool name, algorithm, XY positions, reference Z/Time/channel.

## Tests

Tests are in `tests/test_registration.py` and run inside Docker via:

```bash
./build_workers.sh --build-and-run-tests registration
```

### Test Coverage

| Test | What it verifies |
|------|-----------------|
| `test_interface` | All expected interface keys and algorithm options are present |
| `test_safe_astype_integer` | Integer dtype casting clips values to valid range |
| `test_safe_astype_float` | Float dtype casting works without clipping |
| `test_register_images_control_points_only` | Returns identity matrix when algorithm is "None" |
| `test_register_images_with_algorithm` | Delegates to `sr.register()` for real algorithms |
| `test_compute_single_image_error` | Errors when no IndexRange in tileInfo |
| `test_compute_no_time_dimension_error` | Errors when IndexT is missing |
| `test_compute_no_channels_error` | Errors when no channels are selected |
| `test_compute_basic_functionality` | End-to-end run with Translation algorithm, verifies sink operations |
| `test_compute_with_reference_region` | Verifies `getRegion` is called with bounding box params |
| `test_compute_with_control_points` | Verifies control point annotations are fetched and processed |
| `test_compute_different_algorithms` | Runs Translation, Rigid, and Affine algorithms |
| `test_compute_apply_algorithm_after_control_points` | Both `sr.transform` (matrix calc) and `apply_transform` (output) are called |
| `test_compute_reference_time_adjustment` | Non-zero reference time completes without error |
| `test_compute_metadata_preservation` | Channel names, mm_x, mm_y, magnification are preserved |
| `test_compute_progress_reporting` | `sendProgress` is called during both phases |
| `test_compute_invalid_algorithm_error` | Invalid algorithm name triggers `sendError` |
| `test_compute_reference_region_not_found` | Missing reference region tag triggers `sendError` |
| `test_xy_coordinate_parsing` | XY range string is parsed correctly |
| `test_compute_apply_xy_is_always_list` | `apply_XY` metadata is a JSON-serializable list |
| `test_apply_transform_identity` | Identity matrix returns the original image |
| `test_apply_transform_translation` | Translation matrix shifts the image correctly |
| `test_apply_transform_output_dtype` | Output is always float32 |

### Testing Notes

- All `compute` tests mock `apply_transform` at the `entrypoint` module level to avoid needing real scipy in the output loop.
- Tests that exercise the matrix computation phase (e.g., `test_compute_apply_algorithm_after_control_points`) still mock `sr.transform` on the StackReg instance, since `sr.transform()` is still used there.
- Mock `tileInfo` dicts must include `sizeX` and `sizeY` keys for the auto-crop logic (use `.get()` with default 0, so missing keys are safe but won't trigger auto-crop).

## Lessons Learned

### OOM from pystackreg's Internal float64 Conversion

`pystackreg`'s `StackReg.transform()` and `StackReg.register()` internally cast images to float64 regardless of input dtype. For a 12089x12089 image, this creates ~1.1 GB per array, and multiple temporary arrays during the transform push total memory well past container limits. The fix was twofold: auto-crop for matrix computation (where pystackreg is still used) and replace pystackreg's transform with a float32 scipy equivalent for the output loop.

### Variable Name Collision with `gc`

The entrypoint uses `gc = tileClient.client` (a Girder client instance) on line 432. Importing Python's `gc` module directly would shadow this variable. The import is therefore aliased as `import gc as gc_module`.

### Mock tileInfo Completeness in Tests

The auto-crop logic accesses `tileInfo['sizeX']` and `tileInfo['sizeY']`. Tests that provide minimal mock `tileInfo` dicts (e.g., only `IndexRange`) will KeyError if the code uses direct dict access. Using `.get('sizeX', 0)` in the production code makes it safe for both incomplete mock data and real tile metadata that might lack size info.

## Future TODOs

- **Configurable auto-crop size**: The 2048x2048 center crop is hardcoded. Consider exposing this as an interface parameter or making it adaptive based on available memory.
- **Non-center crop strategies**: The auto-crop always uses the image center, which may not contain the best features for registration. Could allow the user to specify a crop location without requiring a full annotation, or automatically select a region with high feature content.
- **Z-stack registration**: Currently registers across time only. Supporting registration across Z planes (for correcting Z-drift or aligning Z-stacks) would be a natural extension.
- **Streaming output**: The current approach writes the entire registered stack to `/tmp/registered.tiff` before uploading. For very large datasets, a streaming/chunked approach could reduce disk space requirements.
- **Per-frame memory reporting**: Adding memory usage logging (via `psutil` or `/proc/self/status`) would help diagnose future OOM issues before they become crashes.
- **Interpolation order**: `apply_transform` uses bilinear interpolation (`order=1`). Higher-order interpolation (e.g., cubic, `order=3`) could improve quality at a modest memory cost. This could be made configurable.
