# Stitched TIFF Illumination Correction Worker

Corrects grid-locked uneven illumination in an already-stitched microscopy TIFF and uploads a new multi-frame TIFF. It can use a requested algorithm or automatically choose independently for each selected channel, including choosing no correction.

This worker is the stitched-TIFF fallback described by the illumination-correction study. It does not require the original ND2 or a multi-source document. The preferred raw-tile workflow starts from overlapping acquisition tiles and uses overlap/DCT information that is unavailable after compositing; use **Stitch Refinement + Illumination Correction** when those raw sources are available.

## How It Works

1. Loads the requested reference XY/Z/time plane.
2. Fits the physical stitched-tile grid. Automatic reference mode evaluates every available channel, identifies the dominant cross-channel pitch cluster, and uses its highest-quality member. Only the grid geometry is shared across channels.
3. Fits each selected channel independently on the reference Z plane.
4. In automatic algorithm mode, evaluates on representative held-out Z planes:
   - Identity (leave the channel unchanged)
   - BaSiC with darkfield disabled and enabled
   - Folded log-gradient
   - Split-half affine
5. Rejects candidates that damage object-intensity ranking, fine detail, or numeric range, contain non-finite values, or infer an implausible field. Metrics that cannot be measured on a plane are explicitly recorded as unavailable.
6. Finds the Pareto-optimal candidates across the artifact panel and prefers the simpler model inside a fixed 5% tie margin. A correction can displace identity only when it improves the aggregate score by more than that margin and improves every paired held-out Z plane. Spot uniformity participates only for channels explicitly marked punctate.
7. Applies the selected channel model across Z only at the reference XY and reference time, preserves other acquisitions and unselected channels, writes a TIFF, and uploads it to Girder.

Automatic selection is deliberately conservative: if there is no independent Z plane, the identity candidate is returned and the channel is left unchanged with a warning. A manual algorithm can still be requested for a single-plane dataset. Candidate algorithms that are unavailable or fail are reported in a frontend warning and in output metadata. If the identity baseline fails, or every non-identity algorithm fails, the job fails rather than reporting a winner from an unsafe or incomplete comparison.

The workflow is based on the channel-specific evaluation in `FINDINGS_AND_WORKFLOW.md` from the illumination-correction study. The acquisition geometry is estimated once, while flatfield, darkfield, and per-tile gains are never transferred between channels.

## Interface Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| **Channels to correct** | channelCheckboxes | — | Channels for which independent correction models are fitted and applied. |
| **Algorithm** | select | Automatic (recommended) | Choose automatic comparison, BaSiC, folded log-gradient, or split-half affine. |
| **Reference channel mode** | select | Automatically choose best channel | Select the best grid reference from all channels or use the specified channel. |
| **Reference channel** | channel | 0 | Manual grid-reference channel; ignored in automatic reference mode. |
| **Reference XY** | text | blank | 1-based XY used for grid and model fitting; blank uses the current XY. Only this XY is corrected. |
| **Reference Z** | text | blank | 1-based Z used for fitting; blank uses the current Z. Use a well-focused plane. |
| **Reference Time** | text | blank | 1-based time point used for fitting; blank uses the current time. Only this time point is corrected. |
| **BaSiC darkfield** | select | Automatic | In BaSiC mode, compare both settings or force darkfield on/off. Automatic algorithm mode always evaluates both. |
| **Per-tile gain correction** | checkbox | false | Experimental whole-tile gain correction for BaSiC and folded log-gradient. It is estimated from the fit plane and can absorb biological field differences. It has no effect on split-half affine. |
| **Punctate channels for spot metric** | channelCheckboxes | none | Channels for which position-dependent spot counts may influence automatic selection. Leave empty for non-punctate signal. |
| **Output type** | select | Float32 (recommended) | Keep float32 values for audit, or preserve source dtype. Preserve-dtype output fails if more than `1e-4` of corrected pixels would be clipped. |
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

Artifact metrics are calculated against each channel's raw plane. Models are fitted on the requested reference Z and automatic ranking uses representative held-out Z planes (first, middle, and last when available, excluding the fit plane):

- Jitter-aware within-tile amplitude (A1)
- Tile-frequency harmonic modulation (A2)
- Held-out within-tile background dependence (A3)
- Background dynamic range (A5)
- Detrended whole-tile scatter (A6)

The unchanged image is the baseline candidate. A correction must improve the aggregate score by more than the fixed 5% tie margin and improve every paired held-out plane to displace it. This deterministic paired rule avoids treating the two or three correlated validation planes as independent normal samples. Candidates are rejected when any applicable hard preservation guardrail fails:

- Object-intensity Spearman rank below 0.98, when at least 10 measurable objects with nonconstant intensities are available
- Locally normalized high-frequency power below 0.90 of raw, when finite source high-frequency power is available
- Any non-finite source or output pixels
- More than `1e-4` newly nonpositive pixels (pre-existing source zeros are not treated as correction damage)

BaSiC darkfields are also rejected when their mean reaches or exceeds the reference plane's first-percentile image floor. Spot-count uniformity uses a minimum-count requirement and symmetric pseudocounts, and is a soft selection term only for channels marked as punctate.

## Output and Metadata

The worker uploads `/tmp/illumination_corrected.tiff` to the source dataset. Girder metadata records:

- Requested and selected algorithm for every corrected channel
- Full candidate scores, rejection reasons, and unavailable/failed algorithms
- Explicit zero-based and one-based reference channel and XY/Z/time coordinates
- Held-out Z planes, correction scope, pitch bounds, and punctate-metric channels
- Measured pitch, seam positions, residuals, and reference-quality reports
- Model diagnostics, output type, validation setting, worker version, and dependency versions

Channel names, pixel size, and magnification are copied when present in the source tile metadata. Single-frame datasets without `IndexRange` are supported.

## Implementation Notes and Limitations

- This worker is for stitched mosaics with a repeatable physical tile pattern. Each image axis must contain at least four candidate periods and enough detected seams to bound complete tiles.
- Automatic reference selection chooses the best channel on the requested reference XY/Z/time plane; it does not search all Z planes for focus. Navigate to, or specify, a representative focused plane.
- The chosen grid and fitted channel models are shared across Z only at the reference XY and time. Other XY positions and time points are preserved because their acquisition-specific grids and gains may differ.
- Automatic mode needs at least two Z planes. On a single-Z dataset it selects identity; choose a manual algorithm only when fitting and evaluating on the same plane is scientifically acceptable.
- Per-tile gain correction is off by default because it can absorb real whole-tile biology. When enabled, it affects only BaSiC and folded log-gradient.
- Float32 is the recommended audit output. Preserve-source-dtype mode allows only negligible (`<=1e-4`) range clipping and otherwise fails with a request to use Float32.
- Conditional guardrails are recorded as unavailable rather than silently passed. Constant object intensities make Spearman rank undefined and are recorded as unavailable. Non-finite input/output and newly nonpositive-pixel checks are always required.
- Nested grid fitting, model selection, and TIFF writing report progress within monotonic global phases.
- Missing or stale select values are rejected before image loading. The worker reports which setting must be re-selected and raises so Girder records the job as failed rather than successful.
- The worker is CPU-routed (`isGPUWorker=false`). BaSiCPy uses a CPU PyTorch backend, so its image is larger than other classical image-processing workers. The image intentionally overrides BaSiCPy's stale `scipy<1.13` metadata pin while installing its `hyperactive` runtime dependency explicitly. It also pins the matching `pyvips`/libvips wheel used by the TIFF converter and verifies that runtime during the Docker build and tests. Docker tests run real BaSiC fits with darkfield both off and on.

## Build and Test

```bash
./build_workers.sh stitched_illumination_correction
./build_workers.sh --build-and-run-tests stitched_illumination_correction
```
