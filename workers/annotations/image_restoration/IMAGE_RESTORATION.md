# Image Restoration Worker

Restores (denoises/deblurs/deconvolves) fluorescence microscopy images using one of four
reference-free/pretrained algorithms: Noise2Void/N2V2, Cellpose3 restoration, ZS-DeconvNet
(zero-shot), or FluoResFM (pretrained foundation model). The user picks a channel set and
a `Method`; the chosen algorithm is applied per channel and the result is uploaded as a new
multi-frame TIFF, exactly like `histogram_matching`/`deconwolf`/`rolling_ball`.

This worker was designed alongside `illumination_correction` per
[`FLUOR_CORRECTION_WORKERS_SPEC.md`](../../../FLUOR_CORRECTION_WORKERS_SPEC.md) ("Worker 2:
image_restoration"); see that spec for the full design rationale.

## Why these four methods (and not CARE/3D-RCAN)

We usually lack **paired clean ground truth** for fluorescence images, so classic supervised
restoration methods (CARE/CSBDeep, 3D-RCAN) are intentionally **excluded** -- they need a
matched low/high-quality image pair to train on, which most users don't have. All four
methods implemented here work **without** paired references:

| Value | Label | Kind | Dependency | Paired data needed? |
|-------|-------|------|-----------|----------------------|
| `n2v` | Noise2Void / N2V2 (default) | self-supervised, trains on your data | `careamics` (pip, PyTorch) | No |
| `cellpose3` | Cellpose3 restoration | pretrained | `cellpose>=3` (pip) | No |
| `zs_deconvnet` | ZS-DeconvNet | zero-shot, trains on the single input | vendored repo (see below) | No |
| `fluoresfm` | FluoResFM | pretrained, text-prompted foundation model | vendored repo + weights | No |

Time-lapse-specific denoisers (DeepCAD-RT/FAST/TeD) are noted here as **possible future
additions** but are out of scope for this first pass.

## How It Works

1. **Channel selection**: the user selects which channels to restore via `Channels to
   restore` (channelCheckboxes). Unselected channels pass through to the output unchanged.
2. **Method selection**: `Method` picks one of the four algorithms below. The interface
   always shows *all* per-method parameters (there is no conditional UI in this framework);
   each parameter's tooltip states which method it applies to, and parameters for
   non-selected methods are simply ignored by `compute()`.
3. **Per-channel collection**: for each selected channel, every frame of that channel
   (across XY/Z/Time) is gathered into an `(N, Y, X)` stack via `tileClient.getRegion(...)`.
   - `n2v` trains *once* per channel on that channel's whole collection, then predicts on it
     (self-supervised -- no clean target is needed, but more frames generally help).
   - `zs_deconvnet` is zero-shot per frame: it re-trains a small network from scratch on each
     individual frame (no information is shared across frames).
   - `cellpose3` and `fluoresfm` are pretrained and simply applied frame by frame.
4. **GPU/CPU**: `Use GPU` (default on) requests CUDA; `resolve_device()` checks
   `torch.cuda.is_available()` and falls back to CPU with a `sendWarning` if no GPU is
   available (pattern: `deconwolf`'s OpenCL->CPU fallback). CPU is slow for `n2v`,
   `zs_deconvnet`, and `fluoresfm`; `cellpose3` is usually still reasonably fast on CPU.
5. **dtype handling**: restored images are computed in float internally, then
   `np.nan_to_num`'d and clipped/cast back to the source dtype (read from the dataset's
   `dtype` metadata) before being written, to avoid silently bloating a `uint16` source into
   a `float32` output TIFF.
6. **Output assembly**: restored and pass-through frames are reassembled into a
   multi-dimensional TIFF preserving channel names, pixel size (`mm_x`/`mm_y`), and
   magnification, written to `/tmp/restored.tiff` and uploaded back to the dataset's folder.
7. **Provenance**: the uploaded item's metadata records `tool`, `method`, `channels`,
   `gpu_requested`, `device_used`, and the method-specific parameters that were actually used.

## Interface Parameters

| Parameter | Type | Default | Applies to | Description |
|-----------|------|---------|------------|-------------|
| **Method** | select | `n2v` | all | Restoration algorithm: `n2v`, `cellpose3`, `zs_deconvnet`, `fluoresfm`. |
| **Channels to restore** | channelCheckboxes | -- | all | Channels to process; unselected channels pass through unchanged. |
| **Use GPU** | checkbox | true | all | Use CUDA if available; falls back to CPU with a warning otherwise. |
| **Epochs** | number | 20 | n2v | Self-supervised training epochs on the per-channel collection. |
| **Use N2V2** | checkbox | true | n2v | Use the N2V2 variant (reduces checkerboard artifacts vs. classic N2V). |
| **Patch size** | number | 64 | n2v | Training patch size in pixels (square patches; clamped to image size). |
| **Cellpose3 model** | select | `denoise_cyto3` | cellpose3 | Pretrained restoration checkpoint: `denoise_cyto3`, `deblur_cyto3`, `upsample_cyto3`, `denoise_nuclei`, `deblur_nuclei`, `oneclick_cyto3`. |
| **ZS iterations** | number | 300 | zs_deconvnet | Zero-shot self-supervised training steps, run fresh per frame. |
| **ZS upsampling** | checkbox | false | zs_deconvnet | Runs extra physics-consistency deconvolution refinement instead of the published super-resolution (pixel-grid-changing) mode -- see Implementation Details. |
| **Numerical Aperture (NA)** | number | 0.75 | zs_deconvnet | Used with wavelength/pixel size to build an approximate Gaussian PSF. |
| **Emission Wavelength (nm)** | number | 520 | zs_deconvnet | Used to build the approximate PSF. |
| **Pixel Size XY (nm)** | number | 325 | zs_deconvnet | Used to convert the PSF from physical units to pixels. |
| **FluoResFM backbone** | select | `unet_sd_c` | fluoresfm | Model architecture; **must match the `FLUORESFM_WEIGHTS` checkpoint**. `unet_sd_c` = the real text-conditioned FluoResFM foundation model (uses the prompt + BiomedCLIP embedder). `care`/`dfcan`/`unifmir` = the repo's non-text baseline backbones (prompt ignored). |
| **FluoResFM task** | select | `denoise` | fluoresfm | `denoise`, `deconvolution`, `super-resolution` -- builds the default prompt and selects the `unifmir` task index. |
| **FluoResFM text prompt** | text | (derived from task) | fluoresfm | Free-text prompt (used by `unet_sd_c` only). Left blank, defaults to `"fluorescence microscopy image, <task>"`. |

## Implementation Details

### n2v (Noise2Void / N2V2 via CAREamics)

Uses `careamics.config.create_n2v_configuration(...)` + `careamics.CAREamist` per the
documented API: builds an N2V(2) configuration from `Epochs`/`Use N2V2`/`Patch size`, trains
on the full per-channel `(N, Y, X)` collection (`axes='SYX'`, or `'YX'` for a lone frame), and
predicts on the same collection. Patch size is clamped so it never exceeds the image's own
dimensions.

### cellpose3 (pretrained restoration)

Uses `cellpose.denoise.DenoiseModel(model_type=..., gpu=...)` and calls `.eval(frame,
channels=[0, 0])` per frame, matching the documented Cellpose3 restoration API. Weights for
all six exposed checkpoints are pre-downloaded at Docker build time (see
`download_models.py`) so runtime has no network dependency. **Best used as segmentation
preprocessing**, not for quantitative intensity restoration -- Cellpose3's restoration models
are trained to produce clean-looking inputs for its own segmentation, not photometrically
faithful images.

### zs_deconvnet (zero-shot denoise + deconvolution) -- simplified reimplementation

**Honest scope note**: `restore_zs_deconvnet()` is a **good-faith, simplified
reimplementation** of the zero-shot, self-supervised idea behind ZS-DeconvNet (Qiao et al.,
*Nat. Commun.* 2024; [TristaZeng/ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet)),
**not** a literal port of the vendored repo's training scripts. The upstream repo's Python
pipeline (`Python_MATLAB_Codes/train_inference_python/`) is driven via shell scripts
(`train_demo_2D.sh`, `infer_demo_2D.sh`, etc.) with a CLI surface that could not be verified
end-to-end without executing it in this environment (no GPU/Docker build available here). The
Dockerfile still vendors the official repo, pinned to tag `v1.0` (commit `04d2c21`), for
provenance and as a base for a future, more faithful integration.

What is actually implemented, per frame:

1. **Self-supervised denoising** (`_zs_self_supervised_denoise`): a tiny 3-layer residual CNN
   is trained *from scratch on that single frame* using a
   [Neighbor2Neighbor](https://arxiv.org/abs/2101.02824)-style objective -- each non-overlapping
   2x2 pixel block is split into two "neighbor-subsampled" half-resolution images `g1`, `g2`
   (statistically independent noisy samples of the same underlying signal), and the network is
   trained so `f(g1) ~= g2`. This requires no clean reference and no external training data,
   consistent with ZS-DeconvNet's "trains on the single input" framing, for `ZS iterations`
   steps.
2. **Physics-consistent deconvolution** (`_richardson_lucy_gaussian`): classical
   Richardson-Lucy deconvolution against an **approximate Gaussian PSF** built from `NA`,
   `Emission Wavelength (nm)`, and `Pixel Size XY (nm)` via the Abbe resolution criterion
   (`resolution ~= 0.21 * lambda / NA`), standing in for a full Born-Wolf PSF (c.f.
   deconwolf's `dw_bw`) and for the published network's learned deconvolution stage.

**`ZS upsampling` does not change the output's pixel dimensions.** The published method's
super-resolution mode upsamples the image grid; this worker instead runs more RL iterations
(20 vs. 10) as an in-place sharpening refinement. This is a deliberate constraint, not an
oversight: NimbusImage annotations and pixel-scale calibration are defined against the
original image grid (see CLAUDE.md's coordinate-convention notes), and this worker's contract
is "process channels in place, write back a same-shape TIFF" like every other worker in this
family. True super-resolution upsampling would require a separate resampling/re-registration
step for annotations and is out of scope here.

### fluoresfm (foundation model, experimental)

Vendored from [qiqi-lu/fluoresfm](https://github.com/qiqi-lu/fluoresfm) (pinned tag `v1.0.1`),
the canonical repo for FluoResFM (Lu et al., *Nat. Commun.* 2026), a text-conditioned U-Net
trained across 20+ biological structures for denoising/deconvolution/super-resolution.

**This integration is wired against the repo's *real* inference API** (transcribed from the
repo's own test scripts), replacing an earlier stub that imported non-existent modules
(`methods.fluoresfm`, `packages.text_embedding`) and therefore always failed. The real call
sequence (per `3_0_test_it2i.py`, which the README identifies as "test the FluoResFM model"):

1. **Text embedder** — `models.biomedclip_embedder.BiomedCLIPTextEmbedder(path_json,
   path_bin, context_length=160, device)`; `embedder(prompt)` → conditioning tensor of shape
   `[1, 160, 768]`.
2. **Model** — `models.unet_sd_c.UNetModel(in_channels=1, out_channels=1, channels=320,
   n_res_blocks=1, attention_levels=[0,1,2,3], channel_multipliers=[1,2,4,4], n_heads=8,
   tf_layers=1, d_cond=768, pixel_shuffle=False, scale_factor=4, structural_prompt=False,
   n_tokens=0)` (the exact `params` block from `3_0_test_it2i.py`).
3. **Checkpoint** — `torch.load(path, weights_only=True)["model_state_dict"]`, then
   `utils.optim.on_load_checkpoint(...)` to strip the `_orig_mod.` prefix that `torch.compile`
   adds, then `model.load_state_dict(...)`; `.to(device).eval()`.
4. **Per frame** — clip negatives, percentile-normalize (0.03 / 0.995, identical math to
   `utils.data.NormalizePercentile` but inlined to avoid that module's heavy `pydicom`/`scipy`/
   `PSFmodels` import chain), reflect-pad H/W up to a multiple of 8, run
   `model(x, None, text_embed)` (forward signature is `(x, time_steps, cond)`; `time_steps` is
   unused), crop the padding off, then `_rescale_to_reference()` back to the source frame's
   range.

**Backbones (`FluoResFM backbone`).** The repo also ships three *non-text* baseline backbones
(`models.care.CARE`, `models.dfcan.DFCAN`, `models.unifmir.UniModel`, per `3_1_test_i2i.py` —
the README's "test the model **without inputing the text**"). These are exposed for
completeness and are built/loaded/run faithfully (dfcan/unifmir use `scale_factor=1`/`srscale=1`
so the output grid matches the input; `unifmir` takes the task index sr=1/dn=2/iso=3/dcv=4;
`care` is reflect-padded to a multiple of 4). **They ignore the text prompt.** The default and
recommended backbone is `unet_sd_c` — the genuine text-conditioned FluoResFM. **The selected
backbone must match the architecture the `FLUORESFM_WEIGHTS` checkpoint was trained with**
(the repo ships a separate checkpoint per backbone); a mismatch surfaces as a structured
`load_state_dict` error rather than a crash.

**Simplifications vs. the upstream script (honest scope):**
- **No sliding-window patch-stitcher.** The repo tiles large images via
  `utils.data.Patch_stitcher` with a linear-ramp blend; this worker instead runs each frame
  through the model whole (mirroring the repo's own non-patched branch, which triggers when the
  image is smaller than the patch size). Large frames therefore need enough GPU memory to
  process at full size.
- **Free-text prompt is out-of-distribution.** FluoResFM was trained on highly *structured*
  prompts (`"Task: …; sample: …; structure: …; input microscope: …; input pixel size: …;
  target microscope: …; target pixel size: …."`). A short free-text prompt still embeds and
  runs, but for best results supply a prompt in that structured form.
- **No super-resolution grid change.** As with the other methods, output pixel dimensions are
  kept identical to the input so NimbusImage annotations/scale stay aligned.

**Weights are not baked into the Docker image** (they are large and distributed via Google
Drive / Baidu Yun, not a stable scriptable URL). Two runtime artifacts are needed for the
text-conditioned backbone:

| Env var | Contents | Source |
|---------|----------|--------|
| `FLUORESFM_WEIGHTS` | the FluoResFM `.pt` checkpoint (per chosen backbone) | fluoresfm README's Google Drive / Baidu Yun links |
| `FLUORESFM_EMBEDDER_DIR` | dir with `open_clip_config.json` + `open_clip_pytorch_model.bin` (~0.4 GB) | HuggingFace `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (repo's `1_1_download_embedder.py`) |

`resolve_fluoresfm_weights()` and `resolve_fluoresfm_embedder()` read these; if either is
missing, `restore_fluoresfm()` returns `None` after an actionable `sendError` (naming the real
modules/paths) instead of crashing. `download_models.py` still *attempts* a best-effort
checkpoint download via the optional `FLUORESFM_WEIGHTS_URL` build arg (a mirror you control)
and never fails the build if unset. Python deps for this path (`open_clip_torch`,
`transformers`, `huggingface_hub`, `torchinfo`, `timm`) are installed in both Dockerfiles;
`flash-attn` is **not** required (the vendored attention falls back to plain attention and only
uses flash for self-attention, never the text cross-attention).

**Experimental / untested in CI.** This path requires a GPU plus large runtime weights, so it
is **not exercised by the test suite** (the tests mock `restore_fluoresfm` and only verify the
interface, dispatch, weight/embedder resolvers, and the honest error paths). Restored
intensities from a foundation model may be partially hallucinated — follow the general
guidance below before trusting quantitative results from `fluoresfm` or `zs_deconvnet`.

### dtype and NaN handling

All four methods return float arrays internally. `_clip_to_dtype()` runs `np.nan_to_num`
(zeroing any NaN/inf the models might produce) and, for integer source dtypes, clips to the
dtype's representable range before casting -- so a `uint16` input never silently becomes a
`float32` output, and a model excursion above 65535 or below 0 doesn't wrap around.

### Testability / lazy imports

Per the project convention (see CLAUDE.md and `histogram_matching`), every heavy third-party
import (`torch`, `careamics`, `cellpose`, the vendored zs_deconvnet/fluoresfm modules) is
lazy -- imported *inside* the relevant `restore_*` function, never at module load time. This
keeps `interface()` fast and lets `tests/test_image_restoration.py` run natively, with none of
those packages installed, by patching `entrypoint.restore_n2v` / `restore_cellpose3` /
`restore_zs_deconvnet` / `restore_fluoresfm` directly. `compute()`'s dispatch table is resolved
via `globals()` *inside* `compute()` (not bound to function objects at import time), so those
patches take effect correctly.

## Notes

- **GPU strongly recommended** for `n2v`, `zs_deconvnet`, and `fluoresfm`; `cellpose3` is
  comparatively fast even on CPU. The production Dockerfile is built on
  `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`; `Dockerfile_M1` provides a CPU-only build
  (from `nimbusimage/image-processing-base:latest`) for Mac development, with the same
  restoration dependencies (CPU PyTorch instead of the CUDA wheel).
- **`zs_deconvnet` and `fluoresfm` are the highest-risk / most experimental methods** in this
  worker -- see the honest scope notes above. Validate their output before any quantitative
  use.
- **Segment on restored, measure on illumination-corrected raw.** All four restoration
  methods can alter absolute intensities (self-supervised denoising suppresses noise
  non-uniformly; pretrained/foundation models can hallucinate detail). The recommended
  workflow is: use the restored image to find/segment objects (it's usually much easier to
  see structure), but compute quantitative intensity properties (e.g. via the
  `blob_intensity`/`point_circle_intensity` property workers) on the original or
  illumination-corrected (see the sibling `illumination_correction` worker) raw image, not on
  the restored one.
- Cellpose3 checkpoints are pre-downloaded at build time (`download_models.py`); ZS-DeconvNet
  and FluoResFM repos are vendored via pinned `git clone`s in the Dockerfile.
