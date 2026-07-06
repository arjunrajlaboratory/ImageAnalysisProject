# Illumination Correction Worker

This worker corrects spatial illumination artifacts in fluorescence microscopy images --
shading, vignetting, background/dark bias, and scanner/tiling stripes -- using a single
`Method` dropdown that selects between six algorithms: **BaSiC**, a **CIDRE-style**
retrospective estimator, a **CellProfiler-style** illumination function, classical
**flat/dark-field** reference correction, **destripe** (wavelet-FFT stripe removal), and
**SSCOR** (deep-learning stripe self-correction). An optional lightweight QC report can be
generated after correction.

## How It Works

1. **Channel selection**: The user selects which channels to correct via "Channels to
   correct" (a `channelCheckboxes` field). Unselected channels pass through to the
   output completely unchanged.
2. **Collection gathering**: For each selected channel, the worker gathers every frame
   of that channel across XY / Z / Time into a `(N, Y, X)` stack ("collection"). This is
   the input to the retrospective methods (`basic`, `cidre`, `cellprofiler`); the
   reference-based (`flatfield`) and per-frame classical (`destripe`) methods also
   receive this same stack but process each frame independently.
3. **Correction**: `compute()` dispatches on `Method` to one standalone function
   (`correct_basic`, `correct_cidre`, `correct_cellprofiler`, `correct_flatfield`,
   `correct_destripe`, `correct_sscor`), each with the signature
   `f(stack, opts) -> (corrected_stack, diagnostics)`. `sscor` is the one exception to the
   "no upfront validation" rule below the fold: since it requires an externally-supplied
   checkpoint, `compute()` resolves `SSCOR_WEIGHTS` (via `resolve_sscor_checkpoint`) and
   GPU availability *before* the channel loop, `sendError`-and-returns if no checkpoint is
   available, and injects the resolved path/GPU choice into `method_opts` -- mirroring how
   the `flatfield` method's flat-reference check works.
4. **QC (optional)**: If "Report correction quality (QC)" is enabled, `compute_qc_metrics`
   computes a few simple flat-field-quality numbers on the corrected collection for each
   selected channel.
5. **dtype handling**: Corrected values are `np.nan_to_num`'d and, if the source dtype is
   known (`tileClient.tiles['dtype']`), clipped and cast back to that dtype (e.g.
   `uint16`) before being written, to avoid silently bloating the output TIFF to float.
6. **Output assembly**: A `large_image` sink is built frame-by-frame -- corrected frames
   for selected channels, untouched frames for everything else -- preserving the original
   `channelNames` / `mm_x` / `mm_y` / `magnification`, written to
   `/tmp/illumination_corrected.tiff`, and uploaded back to Girder with a provenance
   metadata dict (`tool`, `method`, `channels`, method-specific params, per-channel
   `diagnostics`, and `qc` if requested).

There is no conditional UI in this framework, so **all** parameters for **all** methods
are always shown; each parameter's tooltip states which method(s) it applies to, and
unused parameters for the currently-selected method are simply ignored.

## Methods

| Value | Label | Kind | Dependency | Needs reference frames? |
|-------|-------|------|-----------|--------------------------|
| `basic` | BaSiC (shading + darkfield) | retrospective | `basicpy` (BaSiCPy 2.x, PyTorch) | no |
| `cidre` | CIDRE-style retrospective | retrospective | in-house numpy/scipy | no |
| `cellprofiler` | CellProfiler-style illumination function | retrospective or per-image | in-house numpy/scipy | no |
| `flatfield` | Flat/Dark-field (reference-based) | reference | in-house numpy | **yes** |
| `destripe` | Stripe / tiling-seam correction | classical destriping | `pystripe` (wavelet-FFT) | no |
| `sscor` | Deep-learning stripe self-correction | pix2pix-style GAN inference | vendored `SSCOR` repo + `SSCOR_WEIGHTS` checkpoint | no (needs a trained checkpoint instead) |

Default `Method` is `basic`.

## Interface Parameters

| Parameter | Type | Default | Applies to | Description |
|-----------|------|---------|------------|-------------|
| **Method** | select | `basic` | all | Illumination-correction algorithm: `basic`, `cidre`, `cellprofiler`, `flatfield`, `destripe`, `sscor`. |
| **Channels to correct** | channelCheckboxes | -- | all | Channels to process; unselected channels pass through unchanged. |
| **Estimate darkfield** | checkbox | `True` | basic | Also estimate a darkfield (offset) term. |
| **Flatfield smoothness** | number (0-100) | `1.0` | basic | Smoothness regularization of the fitted flatfield. |
| **Darkfield smoothness** | number (0-100) | `1.0` | basic | Smoothness regularization of the fitted darkfield. |
| **Correct timelapse baseline drift** | checkbox | `False` | basic | Sort the channel's collection by Time and let BaSiC's transform correct for temporal drift; the per-frame baseline (if produced) is recorded in metadata. |
| **Smoothing sigma** | number (1-2000) | `50` | cidre, cellprofiler | Gaussian smoothing sigma (px) applied to the estimated illumination surface. |
| **Dark quantile** | number (0-0.5) | `0.02` | cidre | Per-pixel quantile used to estimate the dark/offset surface. |
| **CellProfiler mode** | select | `regular` | cellprofiler | `regular` = one illumination function from the across-batch average, applied to every frame; `background` = each frame gets its own smoothed background estimate. |
| **Flat-field XY coordinate** | text | -- | flatfield | 1-indexed XY position of the flat-field reference image. Required for `flatfield`. |
| **Dark-field XY coordinate** | text | -- | flatfield | 1-indexed XY position of the dark-field reference image; empty = use the Dark-field constant instead. |
| **Dark-field constant** | number (0-65535) | `0` | flatfield | Constant dark offset used when no dark-field reference frame is given. |
| **Destripe sigma** | number (1-2000) | `128` | destripe | Band-pass sigma for pystripe's wavelet-FFT stripe removal. |
| **Destripe wavelet** | select | `db3` | destripe | Wavelet family: `db3`, `db5`, `haar`, `sym4`. |
| **Destripe level** | number (0-12) | `0` | destripe | DWT decomposition level (`0` = auto). |
| **SSCOR patch size** | number (1-4096) | `256` | sscor | Sliding-window patch size (px) fed to the generator. |
| **SSCOR offset size** | number (1-4096) | `100` | sscor | Sliding-window step/offset size (px) between patches. |
| **SSCOR repeat** | number (1-5) | `1` | sscor | Number of repeated passes (with shifted offsets) combined via max-projection, per upstream `restore.py`. |
| **SSCOR dark threshold** | number (0-255) | `10` | sscor | 8-bit intensity threshold below which the original (uncorrected) pixel is kept instead of the restored value. |
| **Report correction quality (QC)** | checkbox | `False` | all | Computes lightweight EVEN-style QC metrics per corrected channel and stores them in the output item's metadata. |

## Implementation Details

### Retrospective collection semantics

For `basic`, `cidre`, and `cellprofiler`, the "collection" fit on is every frame of the
selected channel across XY, Z, and Time -- gathered into a `(N, Y, X)` stack, fit once,
then applied (`transform`) to every frame in that stack. If a channel's collection has
fewer than 3 frames, `sendWarning` fires: retrospective illumination estimation is
unreliable with very few images.

### Method notes and honest simplifications

- **`basic`**: Uses BaSiCPy 2.x (`from basicpy import BaSiC`), PyTorch backend, CPU-only
  in this worker (no GPU dependency). `get_darkfield`, `smoothness_flatfield`,
  `smoothness_darkfield` map directly to the corresponding interface parameters;
  `fitting_mode='approximate'`. When "Correct timelapse baseline drift" is enabled, the
  collection is sorted by `IndexT` before fitting/transforming and un-sorted afterward;
  the fitted flatfield/darkfield means (and baseline, if produced) are stored per-channel
  under `diagnostics` in the output item's metadata for provenance.
- **`cidre`**: **This is a lightweight, in-house reimplementation of CIDRE's
  retrospective gain/offset model (Smith et al. 2015, Nature Methods) -- not the
  original MATLAB implementation and its full energy-minimization solve.** The offset
  (dark) surface is estimated as a robust per-pixel low quantile (`Dark quantile`,
  default 2%) across the collection, then heavily Gaussian-smoothed (`Smoothing sigma`).
  The gain (flat) surface is the per-pixel median of `frame - offset`, normalized to
  mean 1 and smoothed the same way, as a spatial-regularization stand-in for CIDRE's
  energy minimization. `corrected = (frame - offset) / gain`.
- **`cellprofiler`**: Reimplements the core of CellProfiler's
  `CorrectIlluminationCalculate` module without a CellProfiler dependency. `regular`
  mode averages the whole collection, Gaussian-smooths it, and normalizes to mean 1 to
  get one illumination function applied to every frame; `background` mode does the same
  per-frame, independently. `corrected = frame / illumination_function`.
- **`flatfield`**: Classical `corrected = (raw - dark) / (flat - dark)`, with the gain
  `(flat - dark)` normalized so its mean is 1 (preserving the original intensity scale).
  The flat/dark references are read at `(that XY, Z=0, T=0, that channel)` for the
  channel currently being corrected. If no dark reference is given, dark is treated as
  the constant `Dark-field constant` (default 0). `flatfield` requires a flat-field
  reference; if `Flat-field XY coordinate` is empty, the worker calls `sendError` and
  exits rather than silently producing a meaningless correction.
- **`destripe`**: Uses `pystripe` (`pystripe.core.filter_streaks`), a wavelet-FFT
  destriping method originally built for light-sheet microscopy, applied independently
  per frame. This is the fast, classical, **no-weights-needed** stripe-removal option --
  prefer it when a trained `sscor` checkpoint isn't available.
- **`sscor`**: Deep-learning stripe self-correction, vendored from
  [`lxxcontinue/SSCOR`](https://github.com/lxxcontinue/SSCOR) (pinned commit
  `985479cd79bcf1359e3d9ba44bacd5f372eb2e60`). See "SSCOR integration details" below.
- **QC report (`compute_qc_metrics`)**: **This is a lightweight quantitative stand-in
  for EVEN** (Nat. Commun. 2026, an ML/LDA-based illumination evaluation-and-optimization
  framework) -- EVEN itself is not vendored here. The QC report computes, per corrected
  channel: `cv_mean_image` (coefficient of variation of a smoothed mean image -- lower is
  flatter), `corner_center_ratio` (mean corner vs. mean center intensity of the mean
  image -- closer to 1.0 means less residual vignetting), and `interframe_cv`
  (coefficient of variation of per-frame mean intensity across the collection). These are
  meant only to let a user quickly compare methods against each other, not as a
  publication-grade quality metric.

### SSCOR integration details

SSCOR ([`lxxcontinue/SSCOR`](https://github.com/lxxcontinue/SSCOR), pinned commit
`985479cd79bcf1359e3d9ba44bacd5f372eb2e60`) is a pytorch-CycleGAN-and-pix2pix-style
codebase with no clean importable inference API, so it is integrated exactly like
`deconwolf` shells out to the `dw` binary: via `subprocess`, driving the repo's CLI script
`restore.py` directly (see `correct_sscor` in `entrypoint.py`).

- **Env-gated weights (`SSCOR_WEIGHTS`)**: SSCOR's trained generator checkpoints are not
  baked into the Docker image -- the upstream repo distributes them via Google Drive, not
  a stable direct-download URL, and producing one requires an image-specific **offline
  self-training stage** (proximity sampling + adversarial training with stripe-orientation
  parameters tuned to the dataset) that this worker does not run. `resolve_sscor_checkpoint`
  reads the `SSCOR_WEIGHTS` environment variable; if unset or not a file, `compute()`
  `sendError`s with actionable instructions (download a checkpoint per the upstream
  README, mount it into the container, set `SSCOR_WEIGHTS` to its path -- a
  `latest_net_G.pth` file -- or choose a different Method) **before the channel loop
  runs**, mirroring the `flatfield` method's upfront flat-reference check. This worker
  therefore only exercises SSCOR's **inference** stage.
- **Subprocess-to-`restore.py` integration**: For each frame, `correct_sscor` writes an
  8-bit RGB PNG to a temp "dataroot" directory, copies the resolved checkpoint to
  `<tmp_checkpoints_dir>/sscor/latest_net_G.pth`, and invokes
  `restore.py --dataroot <dir> --name sscor --model sscor --image_name <file> --offset_size
  <SSCOR offset size> --patch_size <SSCOR patch size> --repeat <SSCOR repeat>
  --dark_threshold <SSCOR dark threshold> --checkpoints_dir <tmp_checkpoints_dir> --gpu_ids
  <0 or -1> --eval` with `cwd` set to the vendored repo (`SSCOR_REPO_PATH`, default
  `/sscor`). The restored image is read back from `<dir>/result/restore-<file>` and
  converted from RGB back to a single intensity channel (mean over channels). A non-zero
  return code raises `RuntimeError` with the captured `stderr`, matching how `deconwolf`
  surfaces `dw` failures.
- **8-bit operation (lossy)**: SSCOR's generator only supports 8-bit RGB I/O (PIL-loaded
  input, `tensor2im`-produced `uint8` output). Each frame is therefore rescaled to `uint8`
  `[0, 255]` via per-frame min/max before being handed to `restore.py`, and the 8-bit
  result is rescaled back to the frame's original `[min, max]` range afterward. **This
  round trip is inherently lossy** (8-bit quantization) -- it is a limitation of SSCOR's
  design, not an artifact of this integration.
- **GPU strongly recommended**: `compute()` resolves GPU availability once per run (a
  lazy, best-effort `torch.cuda.is_available()` check; if `torch` isn't importable, it
  falls back to CPU) and passes `--gpu_ids 0` or `--gpu_ids -1` accordingly. If falling
  back to CPU, `sendWarning` fires once: SSCOR on CPU is very slow.

### Numerical stability safeguard

Every mean-normalized gain/illumination surface (`cidre`'s gain, `cellprofiler`'s
illumination function, `flatfield`'s gain) is floored at 5% of its own mean
(`_MIN_GAIN_FRACTION` in `entrypoint.py`) before dividing. The "real" CIDRE and
CellProfiler algorithms solve a regularized (well-posed) model that cannot collapse to
near-zero gain; this worker's lightweight per-pixel-statistics-plus-smoothing stand-ins
have no such built-in guarantee, so without a floor a handful of near-zero pixels (heavy
vignetting at the extreme corners, or noise-dominated low-signal regions) can amplify
noise to enormous corrected values. The 5% floor bounds local amplification to at most
20x while barely affecting well-behaved regions -- this was verified empirically against
synthetic shading/noise stacks during development.

### dtype and coordinate handling

Corrected stacks are computed in `float64` internally. Before being written to the
output sink, they are passed through `np.nan_to_num` (guarding against NaN/inf from any
of the algorithms) and, if `tileClient.tiles['dtype']` is available, clipped to that
dtype's valid range (`np.iinfo` for integer types) and cast back -- this avoids silently
bloating a 16-bit input into a float64 output TIFF. Frame iteration and the
`large_image` sink assembly follow the same `Index*` -> lowercase-key convention used by
`histogram_matching` / `deconwolf` (`IndexXY` -> `xy`, `IndexZ` -> `z`, etc.).

### Heavy imports are lazy

`basicpy` (inside `correct_basic`), `pystripe` (inside `correct_destripe`), and
`subprocess`/`PIL`/`torch` (inside `correct_sscor` and the upfront GPU-detection check in
`compute()`) are imported **inside** those functions only, never at module top level. This
keeps the `interface()` preview path fast (see `todo/worker-startup-latency.md`) and lets
the test suite run natively without any of those packages installed -- tests patch
`entrypoint.correct_basic` / `entrypoint.correct_cidre` / `entrypoint.correct_sscor` / etc.
directly, mirroring `histogram_matching/tests/test_histogram_matching.py`'s pattern of
patching `entrypoint.match_histograms`. (`entrypoint.resolve_sscor_checkpoint` is likewise
patched directly in tests rather than exercising the real `SSCOR_WEIGHTS` env-var check.)

## Notes

- **No GPU required for five of the six methods.** `basic`, `cidre`, `cellprofiler`,
  `flatfield`, and `destripe` all run on CPU; BaSiCPy's PyTorch backend runs fine without
  CUDA for the collection sizes typical of a single dataset. `sscor` is the exception --
  see below.
- **Small collections**: retrospective methods (`basic`, `cidre`, `cellprofiler`) warn
  when a channel's collection has fewer than 3 frames; results are unreliable in that
  regime regardless of method.
- **`flatfield` requires calibration frames** in the dataset itself (identified by XY
  coordinate); if you don't have dedicated flat/dark calibration images, use `basic`,
  `cidre`, or `cellprofiler` instead.
- **`sscor` requires a trained checkpoint and a GPU is strongly recommended.** Unlike the
  other five methods, `sscor` needs a user-supplied generator checkpoint (`SSCOR_WEIGHTS`,
  see "SSCOR integration details" above) that must be produced offline via SSCOR's
  self-training procedure -- there is no bundled/default checkpoint. If you don't have one,
  use `destripe` (`pystripe`) instead: it needs no weights and is much faster, at the cost
  of being a classical (non-deep-learning) method. Inference also works on CPU but is very
  slow; a GPU is strongly recommended.
- Related workers: `histogram_matching` (post-hoc intensity histogram matching across
  frames), `deconwolf` (deconvolution, a different kind of optical correction),
  `rolling_ball` (simple background subtraction).
