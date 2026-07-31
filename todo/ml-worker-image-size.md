# TODO-003: ML Worker Image Size

**Status:** SAM/SAM2 wave merged (PR #160). Second wave (cellpose ×4, stardist,
condensatenet, piscis ×2) — Dockerfile changes landed, **not yet built or
measured**
**Priority:** Medium
**Related:** [TODO-001 — ML Worker Build Optimization](ml-worker-build-optimization.md)

---

# Wave 1 — SAM / SAM2 (merged, PR #160)

## Problem

Each SAM/SAM2 worker image was 35–40 GB. Investigation traced this to the CUDA
stack being shipped **twice**, plus several caches that were never cleaned.

Estimated per-worker budget before the change (unpacked):

| Item | Est. size |
|---|---|
| `nvidia/cuda:12.1.0-cudnn8-devel` base | ~11.6 GB (5.36 GB compressed, per registry manifest) |
| pip `torch` + bundled `nvidia-*` wheels + `triton` | ~8.5 GB (~4.0 GB of compressed wheels, per PyPI) |
| pip wheel cache (never purged) | ~4 GB — a second copy of the above |
| conda env | ~2.5 GB |
| conda pkgs cache (`conda clean` never run) | ~1.5 GB — a second copy of the env |
| apt layer (`build-essential`, `r-base` listed twice) | ~1.5 GB |
| SAM 2.1 checkpoints ×4 | ~1.5 GB |
| Miniforge base install | ~0.6 GB |
| Full-history git clones | ~0.3 GB |

The core redundancy: `nvidia/cuda:*-cudnn8-devel` provides nvcc, headers, static
libs, cuDNN 8 and cuBLAS, while pip's `torch` ships its *own* cuDNN 9 (857 MB),
cuBLAS (581 MB), cuSPARSE (367 MB), cuSOLVER (338 MB), NCCL (303 MB),
cuSPARSELt (239 MB) and cuFFT (201 MB) inside `site-packages`. PyTorch dlopens
the site-packages copies and never touches the system ones, so nearly all of the
base image's CUDA content was dead weight at runtime.

## What was done

Applied to all 7 SAM/SAM2 workers, both `Dockerfile` and `Dockerfile_M1`:

1. **Real multi-stage build.** The `build` stage keeps the `devel` image so
   sam2's optional `_C` CUDA extension (`csrc/connected_components.cu`) still
   compiles with nvcc; the runtime stage ships on `nvidia/cuda:*-runtime` and
   copies over only the finished conda env, the `annotation_client` subtree, and
   the sam2 source tree (both are `pip install -e`, so they must stay at their
   build-time paths). The previous `FROM base as build` was a no-op — nothing was
   ever copied between stages, so everything stayed in the final image.
2. **`ENV PIP_NO_CACHE_DIR=1`** and **`conda clean --all --yes`**.
3. **Dropped `r-base`** (listed twice, unused by any SAM worker) and the whole
   build toolchain from the runtime stage.
4. **Pruned unused deps** from `environment.yml`: `opencv`, `matplotlib`,
   `onnx`, `onnxruntime`, `pycocotools`. None are imported by any SAM worker;
   `pycocotools` is only reachable via SAM1's `output_mode="coco_rle"`, which
   these workers do not set.
5. **Shallow clones** (`--depth 1`) with `.git` removed.

Also fixed along the way:

- The five `sam2_*/environment.yml` files were byte-identical **except for a
  trailing newline** on `sam2_fewshot_segmentation`. That one byte changed the
  `COPY` hash and forked the entire conda + torch layer chain, so five workers
  that could share ~20 GB of layers each carried their own copy. They are now
  identical.
- `sam2_refine/Dockerfile_M1` was a copy-paste of `sam2_propagate` (wrong
  `COPY` paths, wrong labels) — the bug noted in TODO-001. Fixed.
- `sam_automatic_mask_generator/Dockerfile` mixed build contexts: some `COPY`
  lines assumed the worker directory, others assumed the repo root. Normalized
  to repo root, matching every other worker. It also cloned the whole
  `ImageAnalysisProject` repo just to install `annotation_utilities`; it now
  copies the local directory like its siblings.
- The two SAM test images (`tests/Dockerfile_Test`) invoked `conda run`, which no
  longer exists in the runtime image (the base conda install is build-time only
  now; `run_worker.sh` activates the env by path). They now call `pip`/`python`
  directly via `run_worker.sh`.

## Follow-up from PR #160 review

Codex flagged (P1) that `sam_fewshot_segmentation/Dockerfile_M1` installed torch
from `https://download.pytorch.org/whl/cu118`, which publishes no Linux aarch64
wheels — a native arm64 build would fail with "No matching distribution found".
Correct, and it was introduced by this change: the original M1 file had no torch
install line at all. The arm64 variant now takes the plain PyPI (CPU) wheels,
which do ship `manylinux_2_28_aarch64` builds, matching the CPU-only intent of
`MAC_DEVELOPMENT_MODE`.

Worth noting the pre-existing state was also broken, just later: without that
line and with no `pytorch` in `environment.yml`, the M1 image had no torch at
all (`segment-anything` declares no `install_requires`), so it would have failed
at import time rather than build time. The fix resolves both.

Checked at the same time: all four CUDA tags in use
(`11.8.0`/`12.1.0` × `cudnn8-devel`/`runtime`, ubuntu22.04) publish arm64
manifests, so the devel→runtime swap is safe on M1.

## Not yet verified

**None of this has been built.** The changes were made in an environment without
a Docker daemon, so the expected ~10–12 GB result is an estimate, not a
measurement. Before merging:

1. `./build_machine_learning_workers.sh` (or a single `docker build`) and record
   `docker images` sizes before/after.
2. Confirm the sam2 `_C` extension survives the stage copy:
   `python -c "from sam2 import _C; print(_C.__file__)"` inside the image, and
   `ldd` it to confirm it resolves against the torch wheels' `libcudart`.
   This matters most for `sam2_video`, which is the only worker that reaches
   `_C` at runtime — it calls `build_sam2_video_predictor` without
   `apply_postprocessing=False`, so `fill_hole_area=8` and
   `fill_holes_in_mask_scores` runs. The other four pass
   `apply_postprocessing=False` and never touch `_C`.
   Note `sam2_video` builds from the `segment-anything-2-nimbus` fork
   (branch `nimbus-video-predictor`), which was not read during this analysis.
3. Run a real segmentation job per worker to confirm GPU passthrough still
   works from the `runtime` base.

## Remaining opportunities

- **`nvidia/cuda:*-base` instead of `*-runtime`** for the final stage: another
  ~3 GB. Not taken here because `-runtime` guarantees `libcudart` is present for
  `_C.so` without depending on torch having preloaded it first. Worth revisiting
  once the `ldd` check above is done.
- **Trim the checkpoint set.** `download_ckpts.sh` fetches all four SAM 2.1
  checkpoints (~1.5 GB); `sam2.1_hiera_large.pt` alone is ~860 MB. The UI
  defaults to `small`/`base_plus`, but the model list is built by listing the
  checkpoints directory, so dropping one removes it from the user's choices.
- **A shared `sam2-worker-base` image.** Now that the environment files are
  identical this is much more tractable, and would cut aggregate registry/pull
  cost from 5 × full to 1 × base + 5 × small. TODO-001 recorded this as blocked
  on GPU passthrough; that diagnosis is worth re-testing, since Docker labels and
  `ENV` *are* inherited by child images and `com.nvidia.volumes.needed` is a
  legacy nvidia-docker v1 mechanism the modern Container Toolkit ignores. The
  likelier culprit is that `cuda-ml-worker-base` was built `FROM ubuntu:jammy`
  (as `Dockerfile.worker_base` still is), which genuinely has no CUDA in it.
- **Pin `torch`.** Nothing pins it — `pip install -e /code/sam2` resolves to
  whatever is newest on PyPI, which is a size *and* reproducibility hazard.
- **The same pattern applies to the other GPU workers** (cellpose, cellposesam,
  stardist, condensatenet, piscis, deconwolf), which share the `devel`-base and
  `r-base` copy-paste lineage. → done for all but `deconwolf`, see wave 2 below.

---

# Wave 2 — cellpose ×4, stardist, condensatenet, piscis ×2

## Scope

The eight remaining images built by `build_machine_learning_workers.sh`:

| Worker | Image |
|---|---|
| `cellpose` | `annotations/cellpose_worker` |
| `cellpose_train` | `annotations/cellpose_train_worker` |
| `cellposesam` | `annotations/cellposesam_worker` |
| `cellposesam_train` | `annotations/cellposesam_train_worker` |
| `stardist` | `annotations/stardist_worker` |
| `condensatenet` | `annotations/condensatenet` |
| `piscis/predict` | `annotations/piscis_predict` |
| `piscis/train` | `annotations/piscis_train` |

All eight carried the exact lineage wave 1 diagnosed: a `*-devel` CUDA base, a
no-op `FROM base as build`, no `PIP_NO_CACHE_DIR`, no `conda clean`, full-history
git clones, and the copy-pasted `r-base` (listed **twice**) +
`software-properties-common` + `python3` apt block that no worker uses.

## What was done

Same shape as wave 1, per worker:

1. **Real multi-stage build.** The `build` stage keeps the `devel` image (it is
   the only compiler available for any source-only pip dependency); the runtime
   stage ships on `*-runtime` and copies over the finished conda env, the trees
   that were `pip install -e`'d, and the baked-in model cache. Worker `.py`
   files are copied into the runtime stage directly from the build context.
2. **`ENV PIP_NO_CACHE_DIR=1`** and **`conda clean --all --yes`**.
3. **Dropped `r-base` ×2, `software-properties-common`, `python3-software-properties`
   and `python3`** — nothing in these workers uses R or the system interpreter
   (`run_worker.sh` execs the conda env's python directly).
4. **Shallow clones** (`--depth 1`) with `.git` removed.
5. **Pruned the DeepTile checkout** (cellpose, cellposesam, stardist,
   condensatenet). The clone is ~140 MB but the importable package is 160 KB:
   the rest is `.git` (50 MB), sample `data` (56 MB) and `notebooks` (37 MB).
   It is a `pip install -e`, so the tree has to survive into the runtime stage —
   `data`/`notebooks`/`tests` are deleted right after the clone instead.
   Verified safe: DeepTile pins a static `version = "2.0.10"` in
   `pyproject.toml`, so nothing needs `.git` at install time. (Same check for
   `zjniu/Piscis`, which pins `version = '1.1.0'`.)

Unlike wave 1, **no `environment.yml` was touched**: none of these workers had
a dependency that was provably unused, and the two cellpose pairs / the two
piscis images intentionally differ.

### Per-worker specifics

- **cellpose, cellpose_train, cellposesam, cellposesam_train** — runtime base is
  plain `nvidia/cuda:11.8.0-runtime-ubuntu22.04`, i.e. the `cudnn8` tag is
  dropped as well. cellpose takes torch from PyPI, whose wheel bundles its own
  cuDNN / cuBLAS / cuFFT / NCCL under `site-packages` and dlopens those, so the
  image's copy was never loaded. Model cache: `/root/.cellpose` — both cellpose
  3.x and 4.x resolve checkpoints from `~/.cellpose/models`
  (`models.MODEL_DIR`), which is where `download_models.py` writes. The
  `.cellposesam` directory the two Cellpose-SAM workers use for Girder-synced
  custom models is created at runtime and is deliberately *not* baked in.
- **stardist** — the one worker that **keeps** `cudnn8` on the runtime stage.
  TensorFlow 2.11 predates the `tensorflow[and-cuda]` extra (added in 2.14), so
  its wheel declares no `nvidia-*` dependencies and dlopens `libcudnn.so.8`,
  `libcublas`, `libcufft` from the *image*. A plain `-runtime` tag would not
  fail the build: TF logs "Could not load dynamic library libcudnn.so.8" and
  silently falls back to CPU. Model cache: `/root/.keras` (csbdeep's
  `from_pretrained` → `keras.utils.get_file` → `~/.keras/models`).
- **condensatenet** — runtime base `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
  (torch, so no `cudnn` tag needed). Model cache: `/models`, which is
  self-contained — `download_model()` calls `snapshot_download(...,
  local_dir_use_symlinks=False)`, so those are real files, not links into
  `~/.cache/huggingface`.
- **piscis (predict + train)** — runtime base
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04`. Model cache: `/root/.piscis`
  (`piscis.paths.MODELS_DIR`); both workers also write user/trained models there
  at runtime. Two extra fixes here:
  - Both images ran `git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/`
    and installed `annotation_utilities` / `worker_client` **from that clone**,
    so the build used whatever was on the default branch rather than the tree
    being built — the same bug wave 1 fixed in `sam_automatic_mask_generator`.
    They now `COPY ./annotation_utilities` and `./worker_client` like every
    other worker.
  - Miniconda → Miniforge, matching every other worker. That removes the
    `conda tos accept --channel https://repo.anaconda.com/pkgs/{main,r}` calls
    and the `defaults` channel entirely. Safe here because
    `piscis/environment.yml` is only `python=3.11` + `pip`.

## Not yet verified

**None of this has been built** — same constraint as wave 1 (no Docker daemon in
the environment where the change was made). Before merging:

1. `./build_machine_learning_workers.sh` and record `docker images` sizes
   before/after for the eight images above.
2. Per worker, confirm the model cache survived the stage copy and that no
   download happens on first run:
   - cellpose ×4: `ls /root/.cellpose/models` → `cyto/cyto2/cyto3/nuclei`
     (cellpose 3) or `cpsam_v2`/`cpsam` (cellpose 4).
   - stardist: `ls /root/.keras/models` → `2D_versatile_fluo`,
     `2D_versatile_he`.
   - condensatenet: `ls /models/condensatenet /models/condensatenet-v1` →
     `config.json` + `model.safetensors` in each.
   - piscis ×2: `ls /root/.piscis/models` → the four dated models plus the
     `rajlab/raj-lab-piscis-models` collection.
3. Confirm the GPU is actually used from the `runtime` base — not just that the
   job completes. `torch.cuda.is_available()` for the seven torch workers, and
   for stardist `tf.config.list_physical_devices('GPU')` **plus** a check that
   the log has no "Could not load dynamic library libcudnn" line, since TF
   degrades to CPU silently.
4. Run one real job per worker (segmentation for cellpose/cellposesam/stardist/
   condensatenet, spot detection for piscis predict, and a short retrain for the
   three training workers, which is the path that writes into the copied model
   directories).

## Remaining opportunities (wave 2)

- **`deconwolf` is the last GPU worker on this lineage** and was left out of
  this pass: it is an image-processing worker rather than an ML one, and it is
  the only one that compiles a native binary (`cmake` build of `elgw/deconwolf`
  against fftw/gsl/png/tiff + the OpenCL loader). Splitting it needs the runtime
  stage to install the non-`-dev` runtime libs and copy `/usr/bin/dw`, and its
  `tests/Dockerfile_Test` still calls `conda run`, which would have to move to
  `run_worker.sh` the way the two SAM test images did. It should be a bigger
  win than most (it drops `cmake` + `build-essential` + six `-dev` packages),
  and it does not need the CUDA *runtime* libs at all — it reaches the GPU via
  `libnvidia-opencl.so.1`, which the Container Toolkit mounts from the host.
- **`nvidia/cuda:*-base` for the torch workers.** None of them link against the
  image's `libcudart` at all (torch loads its own), so unlike the SAM workers
  there is no `_C.so` argument for keeping `-runtime`. Worth another ~1.5–2 GB
  each, and it is a one-word change once wave 2 is measured.
- **`condensatenet` has no `Dockerfile_M1`**, but
  `build_machine_learning_workers.sh` passes `$DOCKERFILE` (= `Dockerfile_M1` on
  arm64) for it. Pre-existing; an arm64 run of that script fails on this worker.
- **Pin `torch` / `tensorflow`** in the cellpose and piscis environments for the
  same reproducibility reason recorded in wave 1.
