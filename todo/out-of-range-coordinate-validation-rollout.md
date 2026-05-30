# TODO-002: Roll out out-of-range coordinate validation to remaining workers

**Status:** Deferred
**Priority:** Medium
**Related branch/PR:** `fix-batch-coordinate-out-of-range`
**Related issue:** [NimbusImage#1185](https://github.com/arjunrajlaboratory/NimbusImage/issues/1185)
**Design spec:** [docs/superpowers/specs/2026-05-30-out-of-range-coordinate-validation-design.md](../docs/superpowers/specs/2026-05-30-out-of-range-coordinate-validation-design.md)

## Summary

When a user enters a `Batch XY`, `Batch Z`, or `Batch Time` range that includes
coordinates not present in the dataset, workers crash with a bare `KeyError`
from `coordinatesToFrameIndex` (`self.map[channel][T][Z][XY]`) instead of an
actionable message.

The initial fix (this branch) added a shared, pure validator
(`annotation_utilities.coordinate_validation.find_out_of_range` /
`format_out_of_range_message`) and wired it into `WorkerClient.process()` via a
new `WorkerClient.validate_coordinates()`. That covers **only the workers on the
`WorkerClient.process()` path**:

- cellposesam (also gets an early `validate_coordinates()` call before model load)
- cellpose
- condensatenet
- laplacian_of_gaussian
- random_squares
- sample_interface

This TODO tracks extending the same protection to the workers that parse batch
ranges and/or call `coordinatesToFrameIndex()` **directly**, which still crash
with a bare `KeyError`.

## Remaining work — workers that need individual fixes

These parse batch ranges (`process_range_list` / `get_batch_information`) and
iterate `coordinatesToFrameIndex()` / `getRegion()` themselves. They should call
`coordinate_validation.find_out_of_range` + `format_out_of_range_message`
(then `sendError` + raise) before their processing loop:

- `workers/annotations/cellori_segmentation/entrypoint.py` (~L86-117)
- `workers/annotations/sam2_automatic_mask_generator/entrypoint.py` (~L86-88, 127)
- `workers/annotations/sam2_refine/entrypoint.py` (~L198-200, 303)
- `workers/annotations/sam2_propagate/entrypoint.py` (~L310-312, 379, 431)
- `workers/annotations/sam2_video/entrypoint.py` (~L217-219)
- `workers/annotations/sam_fewshot_segmentation/entrypoint.py` (~L268-270, 347, 420)
- `workers/annotations/sam2_fewshot_segmentation/entrypoint.py` (~L268-270, 331, 404)
- `workers/annotations/cellpose_train/entrypoint.py` (~L252-258)
- `workers/annotations/histogram_matching/entrypoint.py` (~L122)

Shared helper that several SAM workers funnel through (fixing it here protects
all its callers):

- `annotation_utilities/annotation_utilities/annotation_tools.py` →
  `get_images_for_all_channels()` (~L213-225) calls `coordinatesToFrameIndex()`.

## Already validate (no change needed — and good reference patterns)

- `workers/annotations/registration/entrypoint.py` — validates reference Z/Time
  against `IndexRange`; intersects parsed batch ranges with valid ranges.
- `workers/annotations/crop/entrypoint.py` — intersects all three batch ranges
  with the dataset ranges.
- `workers/properties/blobs/blob_intensity_worker/entrypoint.py` and
  `blob_annulus_intensity_worker` — build `range_z = range(0, IndexZ)`, filter
  Z planes, and `sendWarning` for out-of-range. (Note: these use a *filter +
  warn* policy; the WorkerClient fix uses *strict error*. Decide per worker
  whether to keep filter-and-warn or move to strict on rollout.)

## Lower-risk (preview functions — single tile)

These load a single tile in their `interface`/preview path; lower risk but still
vulnerable if a user edits batch params:
random_point, random_point_annotation_M1, annulus_generator_M1, gaussian_blur,
rolling_ball, deepcell, stardist, laplacian_of_gaussian (preview).

## Notes

- Property workers that iterate **annotation-sourced** locations (not user batch
  ranges) are lower risk, since locations come from existing annotations; the
  main exposure there is the channel index, which the UI constrains.
- Defense-in-depth at the lowest level (`coordinatesToFrameIndex` raising a
  descriptive exception) is tracked separately in NimbusImage#1185.
