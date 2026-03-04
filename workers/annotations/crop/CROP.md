# Crop Worker

Subsets a multi-dimensional image by XY position, Z plane, time point, and/or spatial region, producing a smaller image that is uploaded to Girder. Useful for extracting regions of interest or reducing dataset size.

## How It Works

1. Parses user-specified ranges for XY, Z, and Time dimensions (defaults to all if left empty)
2. Optionally retrieves a crop rectangle/polygon annotation to define a spatial subregion
3. Iterates through all frames, keeping only those matching the specified dimension ranges
4. Re-indexes the retained frames to form a contiguous output image
5. Writes the cropped result to a new TIFF and uploads it to the dataset

## Interface Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **XY Range** | text | all | XY positions to retain (e.g., "1-3, 5-8"). 1-indexed. |
| **Z Range** | text | all | Z positions to retain (e.g., "1-3, 5-8"). 1-indexed. |
| **Time Range** | text | all | Time positions to retain (e.g., "1-3, 5-8"). 1-indexed. |
| **Crop Rectangle** | tags | -- | Tag of a rectangle or polygon annotation defining the spatial crop region. Uses the bounding box of the first matching annotation. |

## Implementation Details

- Dimension ranges are 1-indexed in the UI and converted to 0-indexed internally via `batch_argument_parser`. Values outside the image's actual range are silently ignored.
- The output frame indices are re-mapped so they start from 0 and are contiguous. For example, if you keep Z planes 3,5,7, they become 0,1,2 in the output.
- The spatial crop (Crop Rectangle) uses `getRegion` with `left`, `top`, `right`, `bottom` in base pixel units, derived from the bounding box of the first annotation matching the specified tag. Both polygon and rectangle annotation shapes are searched.
- All channels are always retained in the output (channel selection is commented out in the code due to front-end limitations with changing channel definitions).
- If the image has no `frames` metadata (single frame), only the spatial crop is applied.

## Notes

- The Crop Rectangle tag searches both polygon and rectangle annotations. If multiple annotations match, only the first one is used.
- Channel names in the output are preserved from the source image.
- Output metadata includes the tool name and crop coordinates (if a spatial crop was applied).
- This worker is useful for reducing file sizes before running expensive computations, or for isolating specific regions/timepoints of interest.
