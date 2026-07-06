# Spec: Fluorescence Illumination-Correction and Image-Restoration Workers

Status: design spec for two new NimbusImage image-processing workers.
Author: initial design pass; implementation delegated per worker.

## Goal

Add two new **image-processing** workers (category `Image Processing`, live under
`workers/annotations/`, following the `histogram_matching` / `deconwolf` /
`rolling_ball` / `registration` convention of reading all frames, processing,
and uploading a new TIFF to Girder):

1. **`illumination_correction`** — corrects spatial illumination / shading /
   vignetting / background bias / striping. Single worker, user picks the
   algorithm from a `select` dropdown. Implements **all** algorithms from the
   ChatGPT illumination shortlist.
2. **`image_restoration`** — denoises / deblurs / restores. Single worker, user
   picks the algorithm. Implements the best **4** methods, deliberately biased
   toward **reference-free / self-supervised / pretrained** methods (we usually
   have no paired clean ground truth), and **includes FluoResFM** per explicit
   request.

Both workers share the same skeleton as `histogram_matching`/`deconwolf`:
`interface()` + `compute()`, `argparse` main, iterate `tileClient.tiles['frames']`,
build a `large_image` sink, copy metadata (`channelNames`, `mm_x`, `mm_y`,
`magnification`), write `/tmp/<name>.tiff`, `gc.uploadFileToFolder`, then
`gc.addMetadataToItem` with a provenance dict (`tool`, `method`, key params).

---

## Design principles (apply to both workers)

1. **One worker, `Method` `select` dropdown.** The interface exposes ALL
   parameters for ALL methods; each parameter's tooltip notes which method(s)
   it applies to. `compute()` dispatches on `Method` to a per-algorithm
   function. (There is no conditional UI in this framework, so unused params
   for the chosen method are simply ignored — document this.)

2. **Per-channel processing.** A `channelCheckboxes` field ("Channels to
   correct" / "Channels to restore") selects channels. Unselected channels
   pass through **unchanged** (exactly like `histogram_matching` and
   `deconwolf`). Each selected channel is corrected/restored independently.

3. **Retrospective estimators operate on a per-channel image collection.**
   For BaSiC / CIDRE / CellProfiler (illumination) the "collection" is the set
   of all frames of that channel across XY / Z / Time. Gather that stack
   (`N, Y, X`), fit the model on it, then transform each frame. Warn (via
   `sendWarning`) if the collection is very small (e.g. < 3 images) since
   retrospective methods are unreliable then.

4. **Testability is mandatory and shapes the code structure.** Each algorithm
   MUST be a standalone function (e.g. `correct_basic(stack, opts) -> stack`,
   `restore_n2v(stack, opts) -> stack`). Heavy third-party imports (`basicpy`,
   `careamics`, `cellpose`, torch, the FluoResFM/ZS-DeconvNet modules) MUST be
   imported **lazily inside those functions**, never at module top level. This
   keeps the `interface` path fast (see `todo/worker-startup-latency.md`) AND
   lets tests run natively without the heavy deps installed. `compute()`'s
   dispatch, frame iteration, sink assembly, metadata copy, channel filtering,
   and error paths are what the pytest suite exercises — with the per-algorithm
   functions **mocked** (patch `entrypoint.correct_basic`, etc.), mirroring how
   `histogram_matching/tests/test_histogram_matching.py` patches
   `entrypoint.match_histograms`.

5. **Graceful degradation.** If a method's optional dependency or model weights
   are missing at runtime, call `sendError(...)` with a clear, actionable
   message (what's missing, how to enable it) and return — do not crash with a
   raw traceback. GPU methods must fall back to CPU with a `sendWarning` when
   CUDA is unavailable (pattern: `deconwolf`'s OpenCL→CPU fallback, and
   `torch.cuda.is_available()`).

6. **Metadata provenance.** Always `addMetadataToItem` with at least
   `{'tool': <name>, 'method': <method>, 'channels': <list>, ...method params}`.

7. **dtype handling.** Preserve input dtype where reasonable. Many of these
   algorithms work in float internally; when writing back, clip and cast to the
   source dtype (read from `tileClient.tiles.get('dtype')` or the frame's own
   dtype) to avoid silent 16-bit→float bloat in the output TIFF. Document the
   cast. Guard against NaN/inf from the algorithms (`np.nan_to_num`).

8. **Progress.** Call `sendProgress(fraction, title, info)` regularly (per frame
   or per stage) so long ML jobs show life.

---

## Worker 1: `illumination_correction`

Path: `workers/annotations/illumination_correction/`
Interface name: `Illumination Correction`  •  category: `Image Processing`
Default tool name: `Illumination Correction`
Base image: **CPU** — `nimbusimage/image-processing-base:latest`
(none of these methods require a GPU; BaSiCPy runs fine on CPU via torch).
No `Dockerfile_M1` needed unless a dep is x86-only — use the same Dockerfile for both (it inherits the multi-arch base). If a `MAC_DEVELOPMENT_MODE` variant is trivial, skip it; the base is already arch-aware.

### Methods (`Method` select) — implement ALL of these

| Value | Label | Kind | Dependency | Needs ref frames? |
|-------|-------|------|-----------|-------------------|
| `basic` | BaSiC (shading + darkfield, recommended) | retrospective | `basicpy` (pip) | no |
| `cidre` | CIDRE-style retrospective | retrospective | in-house numpy/scipy | no |
| `cellprofiler` | CellProfiler-style illumination function | retrospective or per-image | in-house numpy/scipy/skimage | no |
| `flatfield` | Flat/Dark-field (reference-based) | reference | in-house numpy | **yes** |
| `destripe` | Stripe / tiling-seam correction | classical destriping | `pystripe` (pip) | no |

Default `Method` = `basic`.

Plus an **EVEN-style QC** toggle (see below) — not a correction method, an
optional post-correction quality report.

#### Method details

**`basic` — BaSiC / BaSiCPy** (flagship retrospective, Peng et al. 2017; BaSiCPy 2.x, PyTorch backend).
- `pip install basicpy`. API:
  ```python
  from basicpy import BaSiC
  basic = BaSiC(get_darkfield=<Estimate darkfield>,
                smoothness_flatfield=<Flatfield smoothness>,   # default 1.0
                smoothness_darkfield=<Darkfield smoothness>,   # default 1.0
                fitting_mode='approximate')  # or 'ladmap'
  basic.fit(stack)                # stack: (N, Y, X) float
  corrected = basic.transform(stack)
  # timelapse baseline drift: basic.baseline holds per-image baseline after fit
  ```
- Interface params used: `Estimate darkfield` (checkbox, default True),
  `Flatfield smoothness` (number, default 1.0), `Darkfield smoothness`
  (number, default 1.0), `Correct timelapse baseline drift` (checkbox, default
  False — when True, sort the collection by Time and let BaSiC estimate/subtract
  the temporal baseline; store fitted flatfield/darkfield in item metadata).
- Fit per channel on the full (N,Y,X) collection; `transform` all frames of
  that channel. Save the estimated flatfield (and darkfield) mean values in
  metadata for provenance.

**`cidre` — CIDRE-style retrospective** (Smith et al. 2015, Nature Methods).
There is no maintained pip CIDRE. Implement a faithful **CIDRE-style** in-house
estimator (document clearly it is a lightweight reimplementation of CIDRE's
retrospective gain/offset model, not the original MATLAB code):
- Stack all frames of the channel `(N, Y, X)`.
- Estimate an **offset (dark) surface** `z(x,y)` ≈ robust per-pixel low quantile
  (e.g. 2nd percentile) across the collection, then heavily Gaussian-smoothed.
- Estimate a **gain (flat) surface** `v(x,y)` ≈ per-pixel robust central tendency
  (e.g. per-pixel median of `frame - z`), normalized so `mean(v) == 1`, then
  Gaussian-smoothed with a large sigma (spatial regularization stand-in for
  CIDRE's energy minimization).
- Correct: `corrected = (frame - z) / v`.
- Interface params: reuse a `Smoothing sigma (retrospective)` number
  (default e.g. 0.25 × image width, or a pixel value like 50) controlling the
  Gaussian regularization; `Dark quantile` number (default 0.02).
- Document the simplification honestly in the docs and a code comment.

**`cellprofiler` — CellProfiler-style illumination function**.
Reimplement CellProfiler's `CorrectIlluminationCalculate` core (no CellProfiler
dependency):
- Two modes via a `CellProfiler mode` select: `regular` (across-batch:
  average all frames of the channel, then smooth → illumination function) and
  `background` (per-image: heavily smooth each frame to estimate its own
  background). Default `regular`.
- Smoothing method: large-kernel Gaussian or median (expose
  `Smoothing sigma` reused from above). Rescale the illumination function so its
  mean (or its min, CellProfiler-style) is 1.0, then `corrected = frame / illum`.
- For `regular`, the illumination function is computed once per channel from the
  collection average and applied to every frame; for `background`, computed and
  subtracted/divided per frame.

**`flatfield` — Flat/Dark-field reference-based** (classical, gold standard when
calibration frames exist): `corrected = (raw - dark) / (flat - dark)`, rescaled
to preserve mean intensity.
- Reference frames are supplied by **XY coordinate** (1-indexed text inputs,
  like `histogram_matching`'s reference coords): `Flat-field XY coordinate` and
  `Dark-field XY coordinate` (both text; empty = disabled). Per selected channel,
  the flat/dark reference is the frame at (that XY, Z=0, T=0, that channel). If a
  dark ref is empty, treat dark as 0 (or a scalar `Dark-field constant` number,
  default 0). If flat ref is empty for this method, `sendError` (flatfield is
  meaningless without a flat reference) and return.
- Normalize `flat` so its mean is 1 after dark subtraction to preserve overall
  intensity scale. Clip negatives to 0.

**`destripe` — stripe / tiling-seam correction** (stand-in for SSCOR, which is a
non-packaged DL/GAN method): use **`pystripe`** (`pip install pystripe`,
wavelet-FFT destriping, originally for light-sheet):
  ```python
  from pystripe import core as pystripe_core
  filtered = pystripe_core.filter_streaks(frame, sigma=[sigma1, sigma2],
                                          level=level, wavelet=wavelet)
  ```
- Interface params: `Destripe sigma` (number, default 128 — controls
  band-pass), `Destripe wavelet` (select: `db3`,`db5`,`haar`, default `db3`),
  `Destripe level` (number, default 0 = auto). Applied per frame.
- Docs must state this is a **classical destriping** alternative and that the
  DL method SSCOR is not vendored (no packaged weights).

#### EVEN-style QC toggle (not a correction method)

`Report correction quality (QC)` checkbox (default False). When True, after
correction, compute simple flat-field-quality metrics per selected channel on
the corrected collection and emit them via `sendWarning`/print and store in item
metadata: e.g. coefficient of variation of a smoothed mean image, residual
vignetting (corner-vs-center ratio), inter-frame intensity CV. Document that
full **EVEN** (Nat. Commun. 2026; an ML/LDA evaluation-and-optimization
framework) is not vendored — this is a lightweight quantitative QC stand-in that
lets a user compare methods.

### Interface (`illumination_correction`)

```python
interface = {
  'Method': {'type': 'select',
             'items': ['basic','cidre','cellprofiler','flatfield','destripe'],
             'default': 'basic', 'displayOrder': 0,
             'tooltip': 'Illumination-correction algorithm. basic (BaSiC) recommended when no calibration frames.'},
  'Channels to correct': {'type': 'channelCheckboxes', 'displayOrder': 1,
             'tooltip': 'Process selected channels; others pass through unchanged.'},
  # --- BaSiC ---
  'Estimate darkfield': {'type': 'checkbox', 'default': True, 'displayOrder': 2,
             'tooltip': 'BaSiC: also estimate a darkfield (offset) term.'},
  'Flatfield smoothness': {'type': 'number', 'min': 0, 'max': 100, 'default': 1.0, 'displayOrder': 3,
             'tooltip': 'BaSiC: smoothness regularization of the flatfield.'},
  'Darkfield smoothness': {'type': 'number', 'min': 0, 'max': 100, 'default': 1.0, 'displayOrder': 4,
             'tooltip': 'BaSiC: smoothness regularization of the darkfield.'},
  'Correct timelapse baseline drift': {'type': 'checkbox', 'default': False, 'displayOrder': 5,
             'tooltip': 'BaSiC: estimate and remove temporal background/bleaching drift across Time.'},
  # --- CIDRE / CellProfiler shared ---
  'Smoothing sigma': {'type': 'number', 'min': 1, 'max': 2000, 'default': 50, 'displayOrder': 6,
             'tooltip': 'CIDRE/CellProfiler: Gaussian smoothing sigma (px) for the illumination surface.'},
  'Dark quantile': {'type': 'number', 'min': 0, 'max': 0.5, 'default': 0.02, 'displayOrder': 7,
             'tooltip': 'CIDRE: percentile for the dark/offset surface estimate.'},
  'CellProfiler mode': {'type': 'select', 'items': ['regular','background'], 'default': 'regular', 'displayOrder': 8,
             'tooltip': 'CellProfiler: regular=across-batch average; background=per-image background.'},
  # --- flatfield reference ---
  'Flat-field XY coordinate': {'type': 'text', 'displayOrder': 9,
             'vueAttrs': {...placeholder 'ex. 1'...},
             'tooltip': 'flatfield: 1-indexed XY position of the flat-field reference image.'},
  'Dark-field XY coordinate': {'type': 'text', 'displayOrder': 10,
             'tooltip': 'flatfield: 1-indexed XY position of the dark-field reference (empty = use constant).'},
  'Dark-field constant': {'type': 'number', 'min': 0, 'max': 65535, 'default': 0, 'displayOrder': 11,
             'tooltip': 'flatfield: constant dark offset if no dark reference frame.'},
  # --- destripe ---
  'Destripe sigma': {'type': 'number', 'min': 1, 'max': 2000, 'default': 128, 'displayOrder': 12,
             'tooltip': 'destripe: band-pass sigma for wavelet-FFT stripe removal.'},
  'Destripe wavelet': {'type': 'select', 'items': ['db3','db5','haar','sym4'], 'default': 'db3', 'displayOrder': 13},
  'Destripe level': {'type': 'number', 'min': 0, 'max': 12, 'default': 0, 'displayOrder': 14,
             'tooltip': 'destripe: DWT decomposition level (0 = auto).'},
  # --- QC ---
  'Report correction quality (QC)': {'type': 'checkbox', 'default': False, 'displayOrder': 15},
}
```
(Confirm `select` uses `items` list — check how existing workers e.g. cellposesam
or sample_interface declare `select` options and match that exact schema.)

### Dependencies / Dockerfile (`illumination_correction`)

`environment.yml` (conda-forge based, mirror `histogram_matching/environment.yml`):
```yaml
name: worker
channels: [conda-forge, defaults]
dependencies:
  - python=3.11
  - pip
  - numpy>=2.0
  - scipy
  - scikit-image
  - shapely>=2.0.6
  - libtiff
  - openslide
  - libvips
  - tifffile
  - pip:
    - basicpy
    - pystripe
```
Dockerfile: `FROM nimbusimage/image-processing-base:latest`, `COPY entrypoint.py`,
install `large-image-source-*` wheels (see deconwolf), pip-install
`basicpy`/`pystripe` (if not via env), LABELs (`interfaceName="Illumination Correction"`,
`interfaceCategory="Image Processing"`, `isAnnotationWorker=""`,
`description="Corrects illumination/shading/striping (BaSiC, CIDRE, CellProfiler, flat-field, destripe)"`),
and the `run_worker.sh` entrypoint. **Note:** verify `basicpy` installs cleanly
on the base's Python 3.11; if `basicpy` pins an old jax, prefer BaSiCPy 2.x
(torch). If a conda install of torch is needed, add it to env.

---

## Worker 2: `image_restoration`

Path: `workers/annotations/image_restoration/`
Interface name: `Image Restoration`  •  category: `Image Processing`
Default tool name: `Image Restoration`
Base image: **GPU** — `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04` with the
Miniforge+conda pattern from `deconwolf`/`cellposesam` Dockerfiles, **plus a CPU
fallback at runtime** (`torch.cuda.is_available()` → CPU with `sendWarning`).
Provide a `Dockerfile_M1` (CPU-only, `FROM nimbusimage/image-processing-base`)
for Mac dev, like `deconwolf`.

### Methods (`Method` select) — implement these 4 (all reference-free / pretrained)

| Value | Label | Kind | Dependency | Paired data? |
|-------|-------|------|-----------|--------------|
| `n2v` | Noise2Void / N2V2 (self-supervised denoise) | trains on your data, no clean target | `careamics` (pip, PyTorch) | no |
| `cellpose3` | Cellpose3 restoration (denoise/deblur/upsample) | pretrained | `cellpose>=3` (pip) | no |
| `zs_deconvnet` | ZS-DeconvNet (zero-shot denoise + deconv) | zero-shot, trains on the single input | vendored repo | no |
| `fluoresfm` | FluoResFM (foundation model, pretrained) | pretrained, text-prompted | vendored repo + weights | no |

Default `Method` = `n2v`.

**Rationale for the selection (put in docs):** we usually lack paired clean
ground truth, so supervised CARE/CSBDeep and 3D-RCAN are intentionally excluded.
All four chosen methods work without paired references: N2V/N2V2 is
self-supervised (trains on the noisy data itself), Cellpose3 restoration ships
pretrained models, ZS-DeconvNet is zero-shot (trains on the single input volume),
and FluoResFM is a pretrained foundation model (included per user request; treat
as experimental and validate before quantitative use).

#### Method details

**`n2v` — Noise2Void / N2V2 via CAREamics** (`pip install careamics`).
- API (verify against careamics docs at build time):
  ```python
  from careamics import CAREamist
  from careamics.config import create_n2v_configuration
  cfg = create_n2v_configuration(experiment_name='n2v', data_type='array',
          axes='YX', patch_size=[64,64], batch_size=16,
          num_epochs=<Epochs>, use_n2v2=<Use N2V2>)
  engine = CAREamist(cfg)
  engine.train(train_source=stack)     # self-supervised on the (N,Y,X) stack
  restored = engine.predict(source=stack)
  ```
- Trains **per channel** on that channel's collection (self-supervised).
- Params: `Epochs` (number, default 20), `Use N2V2` (checkbox, default True —
  N2V2 reduces checkerboard artifacts), `Patch size` (number, default 64).
- GPU strongly recommended; CPU fallback allowed but warn it is slow.

**`cellpose3` — Cellpose3 restoration** (`pip install cellpose>=3`).
- API:
  ```python
  from cellpose import denoise
  model = denoise.DenoiseModel(model_type=<Cellpose3 model>, gpu=<gpu>)
  restored = model.eval(frame, channels=[0,0])
  ```
- `Cellpose3 model` select: `denoise_cyto3`, `deblur_cyto3`, `upsample_cyto3`,
  `denoise_nuclei`, `deblur_nuclei`, `oneclick_cyto3` (verify exact model names
  against installed cellpose version; expose a reasonable subset). Default
  `denoise_cyto3`.
- Pretrained; applied per frame. Note in docs: best used as **segmentation
  preprocessing**, not for quantitative intensity restoration. Cellpose weights
  download on first use — do it at build time via a small `download_models.py`
  (pattern: cellposesam) so runtime has no network dependency.

**`zs_deconvnet` — ZS-DeconvNet** (zero-shot; Nat. Commun. 2024;
project https://tristazeng.github.io/ZS-DeconvNet-page/, code on GitHub).
- Vendor by `git clone` in the Dockerfile (pin a commit). ZS-DeconvNet trains a
  small network **on the single input image/stack** (no external training data)
  using its physics-consistency loss, then infers. Wrap its training+inference
  entry so `restore_zs_deconvnet(stack, opts)` runs the zero-shot pipeline per
  frame (2D) or per Z-stack (3D).
- Params: `ZS iterations` (number, default per repo), `ZS upsampling`
  (checkbox — SR mode vs denoise-only), and a PSF spec (reuse NA / wavelength /
  pixel-size numbers like deconwolf, OR let it estimate). Keep params minimal;
  default to denoise+deconv 2D.
- **Highest integration risk.** If the repo's API cannot be driven cleanly,
  implement its published 2D zero-shot training loop directly (dual-stage
  denoise→deconv with the Richardson-Lucy-consistency loss) in a
  `zs_deconvnet.py` module. If truly infeasible in scope, the method must still
  be selectable and `sendError` a clear "ZS-DeconvNet unavailable in this build"
  message rather than crash — but real integration is the goal. Document
  clearly what was implemented.

**`fluoresfm` — FluoResFM** (foundation model; Nat. Commun. 2026; napari plugin
`napari-fluoresfm` by Qiqi Lu; depends on PyTorch + triton; pretrained `.pt`
checkpoint; text-prompt conditioned).
- Vendor the inference code (`git clone` the FluoResFM repo — find the canonical
  repo, likely under the same author as UNiFMIR `github.com/cxm12`; the napari
  plugin `napari-fluoresfm` is the reference). Weights: download the pretrained
  `.pt` at build time via `download_models.py` if a stable URL (Zenodo/HF) is
  known; otherwise load from a path configurable by env var
  `FLUORESFM_WEIGHTS` and, if absent at runtime, `sendError` with instructions
  (where to obtain weights, which env var / mount to set). Do NOT fail the build
  if weights can't be fetched — make weights runtime-optional with a clear error.
- Params: `FluoResFM task` (select: `denoise`,`deconvolution`,`super-resolution`,
  default `denoise`), `FluoResFM text prompt` (text, optional — the model is
  text-conditioned; provide a sensible default prompt describing the structure,
  e.g. "fluorescence microscopy, <task>"), applied per frame.
- Mark **experimental** in docs; warn that restored intensities may be
  hallucinated and should be validated before quantitative use (measure on
  illumination-corrected raw, segment on restored).

### Interface (`image_restoration`)

Mirror the illumination worker's structure: `Method` select (default `n2v`),
`Channels to restore` channelCheckboxes, then the per-method params above, each
tooltip-tagged with its method. Include a `Use GPU` checkbox (default True) with
runtime `torch.cuda.is_available()` fallback + `sendWarning`.

### Dependencies / Dockerfile (`image_restoration`)

GPU Dockerfile modeled on `cellposesam`/`deconwolf`:
- `FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`, Miniforge, `conda env
  create` from `environment.yml`, install NimbusImage annotation_client (clone
  `https://github.com/arjunrajlaboratory/NimbusImage/`, pip install its
  annotation_client), install `annotation_utilities`, install large_image
  wheels.
- pip: `torch` (CUDA build via the cuda base), `careamics`, `cellpose>=3`,
  `triton` (for fluoresfm), `tifffile`. Clone ZS-DeconvNet and FluoResFM repos
  (pinned commits). Run `download_models.py` for cellpose + (best-effort)
  fluoresfm weights.
- LABELs: `isUPennContrastWorker=True`, `isAnnotationWorker=True` (image-processing
  workers use the annotation-worker slot here — match `deconwolf`'s labels which
  use `isAnnotationWorker=""`), `interfaceName="Image Restoration"`,
  `interfaceCategory="Image Processing"`,
  `description="Restores images (Noise2Void/N2V2, Cellpose3, ZS-DeconvNet, FluoResFM)"`,
  `defaultToolName="Image Restoration"`. Include the
  `com.nvidia.volumes.needed`/`NVIDIA_*` env from the GPU workers.
- `run_worker.sh` entrypoint.
- `Dockerfile_M1`: CPU-only `FROM nimbusimage/image-processing-base:latest`,
  same pip installs minus CUDA torch (use CPU torch), no fluoresfm/zs weights
  required to build.

---

## Tests (both workers)

Follow `histogram_matching/tests/` exactly:
- `tests/__init__.py`, `tests/Dockerfile_Test` (same template: `FROM
  annotations/<worker>:latest AS test`, `pip install pytest pytest-mock`, copy
  tests, pytest entrypoint), `tests/test_<worker>.py`.
- Tests run **natively without heavy deps** by mocking:
  - `annotation_client.tiles.UPennContrastDataset` (fixture with `.tiles`,
    `.getRegion`, `.coordinatesToFrameIndex`, `.client`),
  - `large_image.new` (mock sink),
  - `annotation_client.workers.UPennContrastWorkerPreviewClient`,
  - each per-algorithm function (`entrypoint.correct_basic`,
    `entrypoint.correct_cidre`, ... / `entrypoint.restore_n2v`, ...) — patched so
    heavy libs are never imported.
- Required test cases per worker:
  1. `test_interface` — interface set once; asserts `Method` select exists with
     expected items, channel field present, key params present.
  2. Dispatch: for each `Method` value, `compute` calls the right per-algorithm
     function (assert the mock for the selected method is called and others are
     not).
  3. Channel filtering: only selected channels processed; unselected frames
     copied unchanged to the sink.
  4. Output plumbing: `sink.write('/tmp/<name>.tiff')`, `uploadFileToFolder`,
     `addMetadataToItem` with `tool` + `method`.
  5. Metadata preservation: `channelNames`, `mm_x`, `mm_y`, `magnification`.
  6. Error paths: no channels selected → `sendError`; single frame / no `frames`
     → handled; `flatfield` with no flat reference → `sendError`.
  7. Progress reporting emitted.
- Register tests in the registry's "Tests" column (Yes).

## Docs (both workers)

`ILLUMINATION_CORRECTION.md` and `IMAGE_RESTORATION.md` in each worker dir,
following the template in CLAUDE.md / `DECONWOLF.md`: title + 1–2 sentence
description, How It Works, **Interface Parameters table** (every param, type,
default, which method it applies to), Methods table (algorithm, kind, dependency,
reference-free?), Implementation Details (coordinate/dtype handling, retrospective
collection semantics, CIDRE/CellProfiler simplifications, SSCOR→pystripe and
EVEN→QC substitutions honestly noted), Notes (GPU requirements, FluoResFM/ZS
experimental caveats, "segment on restored, measure on illumination-corrected
raw" guidance).

## Registry

Add both workers to `REGISTRY.md` under "Annotation Workers" (that's where the
other Image Processing workers like Deconwolf, Histogram Matching, Rolling Ball
live). Columns: name, description, GPU (Image Restoration = Yes, Illumination
Correction = blank), Tests = Yes, Docs link. Do NOT run
`generate_worker_docs.py --force` (it clobbers hand-written docs); edit
`REGISTRY.md` by hand.

## Out of scope / honest substitutions (state in docs)

- **SSCOR** (DL stripe correction): no packaged weights → `destripe` uses
  classical `pystripe` (wavelet-FFT) instead.
- **EVEN** (ML evaluation/optimization framework): not vendored → optional
  lightweight QC metrics report instead.
- **CIDRE**: no maintained pip package → faithful in-house gain/offset
  reimplementation (documented as such).
- **Supervised restoration** (CARE/CSBDeep, 3D-RCAN) and time-lapse-specific
  denoisers (DeepCAD-RT/FAST/TeD): excluded from the restoration worker because
  they need paired data or are niche; the 4 chosen methods cover the
  reference-free case. Note them as possible future additions.
