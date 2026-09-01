# Stitch Refinement + Illumination Correction Worker

Creates a new corrected pyramidal TIFF from a composited dataset's original raw ND2. It refines tile translations using seeded cross-correlation, applies the supplied raw-tile overlap-DCT flat-field model independently to every channel, and leaves the original image unchanged.

Existing annotations are not transformed. If a dataset already has annotations, their coordinates can shift by tens of pixels relative to the corrected image.

## How It Works

1. Downloads the dataset's `multi-source2.json` and resolves its single referenced ND2 item. The job fails before processing with an actionable error if the dataset contains only an already-stitched TIFF, the source document is missing, or the original ND2 was deleted.
2. Reuses every deployed source transform as the stitch seed. It never re-derives stage geometry and never changes `s11`, `s12`, `s21`, or `s22`.
3. Creates one float32 max-Z reference tile per stage position from the selected channel.
4. Builds the four-neighbor stage grid using the validated 50 µm bin-gap rule.
5. Searches each overlap around its metadata prediction (±24 px at 3 px steps, then ±4 px at 1 px steps), measures NCC, and drops pairs below the selected threshold.
6. Solves all confident translation constraints together with NCC weights and a zero-mean coordinate-shift constraint. Disconnected tiles use a global similarity fit as a conservative fallback.
7. Fits an order-5 smooth DCT flat field per channel from aligned raw-tile overlaps. The recommended mode also fits regularized per-position gains capped to a 1.10-fold range.
8. Corrects every raw P×T×Z×C plane into a scratch BigTIFF, using the ND2's explicit `loop_indices` to preserve position even when its flattened sequence is not P-major. It writes a refined multi-source document pointing to those frames and streams that document through `large_image_converter` into a lossless pyramidal TIFF. The complete mosaic is never materialized in memory.
9. Uploads the pyramidal TIFF as a new item in the dataset folder and records the source IDs, parameters, pair measurements, residuals, bounds, and channel-model diagnostics in Girder metadata.

## Interface Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| **Refine stitch positions** | checkbox | true | Apply the globally solved translations. Clearing it still measures overlaps for DCT correction but retains the exact deployed positions, including fractional translations. |
| **Refinement channel** | channel | 0 | Channel used for max-Z alignment. Channel 0 (the first channel, normally DAPI) is the validated default. |
| **NCC threshold** | number | 0.5 | Reject adjacent pairs below this score. Lower values retain more dim/low-texture pairs but risk false matches; higher values are stricter but can disconnect the grid or reject every pair. Valid range: 0.5–1.0. |
| **Illumination algorithm** | select | Overlap DCT + tile gains (recommended) | Use overlap-DCT with or without small regularized per-position gains. |
| **Output filename** | text | automatic | Optional `.tif`/`.tiff` filename. The automatic name is based on the source ND2. |

## Position Refinement

For a pair `i → j`, the measured content shift is `S`. The placement constraint uses the existing shared camera transform `M`: `p_j - p_i = M S` (`M = -I` for the validated Nikon composites). Search is local and metadata-seeded; there is no blind global registration. Confident constraints are solved jointly, translations are rounded to integers, and the mean coordinate change is held at zero.

The job report contains every predicted/measured offset and NCC, the confident-pair count, per-pair solved residuals, maximum residual, maximum position change, original/output mosaic bounds, and the similarity fallback matrix. A residual above 2 px produces a warning. A second warning reports when any outer mosaic boundary changes by more than the 16 px coordinate-stability target; the measured Well_2 correction is globally self-consistent but accumulates beyond that target, so this condition is reported rather than silently discarding the validated alignment.

Leave the NCC threshold at the validated default of 0.5 for normal runs. Raise it only when the report shows low-NCC pairs with inconsistent offsets, a high residual, or visible misregistration. A higher value rejects those ambiguous matches but can disconnect part of the tile graph; if too few pairs remain, return toward 0.5 or choose a sharper, higher-texture refinement channel. The Well_2 acceptance run did not require tuning: all 84 adjacent pairs had NCC above 0.91.

## Illumination Correction

The implementation follows the supplied raw-tile v7 model:

- Camera-coordinate log-median base field
- Robust Huber IRLS fit of all nonconstant order-5 2-D DCT terms
- Ridge penalty `4 × (1 + order_y² + order_x²)²`
- Six IRLS iterations
- Optional overlap-derived per-position gains with ridge 8 and a 1.10-fold cap

The flat-field reference is the ND2 Z-stack home plane at T=0 (or the middle Z plane when the metadata has no valid home index). Training reads those T=0 P×Z camera frames one at a time and never materializes the full time series; correction still streams and preserves every time point. Fields are fitted at 128×128 and bicubically expanded to the raw camera dimensions. Corrected data is clipped only when written back to lossless uint16.

Each scratch TIFF page records explicit `IndexC`, `IndexT`, and `IndexZ` frame metadata so `large_image` groups positions while preserving the source channel/Z/time axes. The image pins the matching bundled `pyvips`/`libvips` wheel used by `large_image` and asserts the native binding version during its build; conversion is serialized to avoid oversubscribing a CPU worker.

## Output and Metadata

The new item metadata includes:

- `meta.tool` equivalent Girder metadata key `tool`
- Worker version and all interface parameters
- Original ND2 and `multi-source2.json` item IDs
- Zero-based refinement channel and flat-field reference Z
- All pairwise measurements, accepted-pair residuals, coordinate bounds, and final translations
- Per-channel DCT, robust-fit, flat-field, and gain diagnostics

The source `multi-source2.json` and ND2 are read-only. The derived TIFF is independently deletable through the existing large-image dropdown.

## Limitations

- The input must be the original ND2-backed Nimbus multi-source dataset. An already-stitched TIFF does not retain the independent raw tile overlaps required for either position refinement or overlap-DCT fitting and is rejected before processing.
- v1 supports a single ND2 path referenced by one standard `multi-source2.json`. Mixed-source composites are rejected.
- Every ND2 P×T×Z×C frame must be represented exactly once in the source document.
- Source tiles must share one invertible camera transform. Only translations are refined.
- At least one adjacent pair must meet the NCC threshold; otherwise overlap-DCT cannot be fit.
- Scratch storage must accommodate the corrected raw-tile BigTIFF and final pyramidal TIFF simultaneously.
- Existing annotation coordinates are not migrated in v1.

## Build and Test

```bash
./build_workers.sh illumination_correction
./build_workers.sh --build-and-run-tests illumination_correction
```
