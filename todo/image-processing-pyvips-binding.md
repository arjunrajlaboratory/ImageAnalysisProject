# TODO-006: Align image-processing base `pyvips` with its native `libvips`

**Status:** Open; the illumination-correction worker has a local compatibility pin
**Priority:** Medium
**Owner:** unassigned
**Related:** `workers/annotations/illumination_correction/Dockerfile`

## Problem

The image-processing base currently combines a `pyvips` Python binding and
native `libvips` installation that crash inside `large_image_converter` when it
opens a multi-source document. A real converter regression reproduced the
failure as a segmentation fault rather than a Python exception, so a worker can
appear to build correctly and still terminate during conversion.

The illumination-correction worker works around this by installing the exact
bundled wheel `pyvips==3.1.1.8.18.2` from the `large_image` wheel index and
asserting API mode plus native version 8.18.2 during its image build. Its Docker
test converts a small tiled multi-source document end to end.

## Recommended resolution

1. Reproduce the converter smoke test directly in the image-processing base.
2. Pin a mutually compatible `pyvips`/`libvips` pair in the base instead of in
   individual workers.
3. Add the conversion smoke test to base-image validation on both amd64 and
   arm64.
4. Remove the illumination-correction worker's local pin after the fixed base
   has been rebuilt and verified.
