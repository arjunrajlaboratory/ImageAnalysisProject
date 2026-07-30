# TODO-003: SAM / SAM2 Worker Image Size

**Status:** Partially resolved — Dockerfile changes landed, **not yet built or measured**
**Priority:** Medium
**Related:** [TODO-001 — ML Worker Build Optimization](ml-worker-build-optimization.md)

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
  `r-base` copy-paste lineage.
