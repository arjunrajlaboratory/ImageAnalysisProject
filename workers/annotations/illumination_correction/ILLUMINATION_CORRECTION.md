# Illumination Correction Worker

Corrects grid-locked uneven illumination in stitched microscopy images and uploads a new multi-frame TIFF. It can use a requested algorithm or automatically choose a safe method independently for each selected channel.

## How It Works

1. Loads the requested reference XY/Z/time plane.
2. Fits the physical stitched-tile grid. Automatic reference mode evaluates every available channel, identifies the dominant cross-channel pitch cluster, and uses its highest-quality member. Only the grid geometry is shared across channels.
3. Fits each selected channel independently on the reference plane.
4. In automatic algorithm mode, evaluates:
   - BaSiC with darkfield disabled and enabled
   - Folded log-gradient
   - Split-half affine
5. Rejects candidates that damage object-intensity ranking, fine detail, or numeric range, or that infer an implausible field.
6. Ranks valid candidates using the artifact panel plus a soft penalty for position-biased spot detection. When candidates are within 5%, it prefers the simpler model.
7. Applies the selected channel model to every XY, Z, and time frame, preserves unselected channels, writes a TIFF, and uploads it to Girder.

The workflow is based on the channel-specific evaluation in `FINDINGS_AND_WORKFLOW.md` from the illumination-correction study. The acquisition geometry is estimated once, while flatfield, darkfield, and per-tile gains are never transferred between channels.

## Interface Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| **Channels to correct** | channelCheckboxes | — | Channels for which independent correction models are fitted and applied. |
| **Algorithm** | select | Automatic (recommended) | Choose automatic comparison, BaSiC, folded log-gradient, or split-half affine. |
| **Reference channel mode** | select | Automatically choose best channel | Select the best grid reference from all channels or use the specified channel. |
| **Reference channel** | channel | 0 | Manual grid-reference channel; ignored in automatic reference mode. |
| **Reference XY** | text | blank | 1-based XY used for grid and model fitting; blank uses the current XY. |
| **Reference Z** | text | blank | 1-based Z used for fitting; blank uses the current Z. Use a well-focused plane. |
| **Reference Time** | text | blank | 1-based time point used for fitting; blank uses the current time. |
| **BaSiC darkfield** | select | Automatic | In BaSiC mode, compare both settings or force darkfield on/off. Automatic algorithm mode always evaluates both. |
| **Per-tile gain correction** | checkbox | true | Correct residual whole-tile gain variation after estimating the shared within-tile field. |
| **Output type** | select | Float32 (recommended) | Keep float32 values for audit, or clip and cast back to the source dtype. |
| **Validate every corrected plane** | checkbox | true | Recheck object rank, high-frequency detail, and numeric range on every output frame. |
| **Minimum tile pitch** | number | 150 px | Lower bound for physical stitched-tile pitch detection. |
| **Maximum tile pitch** | number | 1400 px | Upper bound for physical stitched-tile pitch detection. |

## Algorithms

### BaSiC

Complete measured seam-to-seam intervals are resampled to a common 256×256 tile coordinate and passed to BaSiCPy 2.0. The inferred field is expanded across the mosaic using the measured, slightly jittered seam locations. When enabled, correction is `(raw - dark) / flat + mean(dark)`, which restores the mean pedestal. Automatic mode tests darkfield both off and on because the additive term is channel-specific.

### Folded log-gradient

The method robustly combines aligned gradients of log intensity across physical tiles, integrates the periodic gradient field with a Fourier-domain Poisson solve, and expands the result over the mosaic. It is useful for dense or saturated channels where a low background quantile is contaminated by biology.

### Split-half affine

This conservative comparator estimates separable multiplicative and additive position-locked curves. Broad-to-fine spatial bands are retained in proportion to their split-half reproducibility, reducing the chance that non-repeating biological structures enter the correction.

## Automatic Evaluation

Artifact metrics are calculated against the channel's raw reference plane:

- Jitter-aware within-tile amplitude (A1)
- Tile-frequency harmonic modulation (A2)
- Held-out within-tile background dependence (A3)
- Background dynamic range (A5)
- Detrended whole-tile scatter (A6)

Candidates are rejected when any hard preservation guardrail fails:

- Object-intensity Spearman rank below 0.98
- Locally normalized high-frequency power below 0.90 of raw
- More than `1e-4` of pixels nonpositive

BaSiC darkfields are also rejected when their mean reaches or exceeds the reference plane's first-percentile image floor. Spot-count uniformity is a soft selection term rather than a hard guardrail because it is not equally meaningful for every channel.

## Output and Metadata

The worker uploads `/tmp/illumination_corrected.tiff` to the source dataset. Girder metadata records:

- Requested and selected algorithm for every corrected channel
- Full candidate scores and rejection reasons
- Reference channel and XY/Z/time coordinates
- Measured pitch, seam positions, residuals, and reference-quality reports
- Model diagnostics, output type, and validation setting

Channel names, pixel size, and magnification are copied when present in the source tile metadata. Single-frame datasets without `IndexRange` are supported.

## Implementation Notes and Limitations

- This worker is for stitched mosaics with a repeatable physical tile pattern. Each image axis must contain at least four candidate periods and enough detected seams to bound complete tiles.
- Automatic reference selection chooses the best channel on the requested reference XY/Z/time plane; it does not search all Z planes for focus. Navigate to, or specify, a representative focused plane.
- The chosen grid and each channel's fitted model are applied across all XY, Z, and time frames. Enable per-plane validation when illumination stability across those dimensions has not already been established.
- Per-tile gain correction is enabled because it improved the study dataset, but it can absorb real field-to-field biology. Disable it when whole-tile biological differences are expected.
- Float32 is the safe default. Source-dtype output clips out-of-range corrected values before casting.
- Missing or stale select values are rejected before image loading. The worker reports which setting must be re-selected and raises so Girder records the job as failed rather than successful.
- The worker is CPU-routed (`isGPUWorker=false`). BaSiCPy uses a CPU PyTorch backend, so its image is larger than other classical image-processing workers.

## Build and Test

```bash
./build_workers.sh illumination_correction
./build_workers.sh --build-and-run-tests illumination_correction
```
