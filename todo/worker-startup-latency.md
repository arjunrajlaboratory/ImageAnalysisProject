# TODO-002: Worker Startup Latency Audit & Slimming Plan

**Status:** In progress
**Priority:** High
**Related PR:** [#148](https://github.com/arjunrajlaboratory/ImageAnalysisProject/pull/148)

## Summary

NimbusImage runs each worker as a fresh container that is torn down per job:

```bash
docker run --rm <image> ... --apiUrl --token --request --parameters --datasetId
```

Images are already **local** (NOT pulled at runtime), so registry pull is not the
cost. The goal of this audit is to reduce the ~1+ second of **in-container
startup** that elapses before real work begins.

Measurements were taken on native **arm64** local builds. Production is **amd64**,
but the relative costs hold.

**Headline:** The `conda run` wrapper in the worker `ENTRYPOINT` is **~70% of
startup cost** and is identical across all 76 worker images. Imports are a
distant second. Base-image / R / size issues do **NOT** affect startup (images
are local, never pulled) — they are disk/build-only concerns.

## Measurements

Light worker `properties/blob_intensity` and ML worker `annotations/deconwolf`:

| What | Time |
|------|------|
| Container + DIRECT env-python boot, no imports | ~0.20 s (floor) |
| `conda run … python -c "pass"` (no imports) | ~1.0–1.2 s → **+0.8–1.0 s pure wrapper overhead** |
| Full entrypoint, all top imports, no network (`--help`) | ~1.2–1.3 s (imports add only ~0.2–0.3 s) |
| micromamba `_entrypoint.sh … python -c pass` (test base) | ~0.22 s (activation with ~0 overhead) |

`deconwolf` (image-processing base, imports `large_image`/`tifffile`) measured
identically: `conda run` ~1.0–1.2 s, imports +0.2–0.3 s.

### Why `conda run` is slow

`conda run` launches a **second** Python process — the `conda` CLI is itself a
Python program that parses args, sources activation, then execs the target
interpreter. That is two interpreter starts plus conda's own imports, per job.

## Section 1 — Entry points

There are 59 `workers/**/entrypoint.py` files, all of identical shape:

1. All top-level imports run.
2. Then `argparse`.
3. Then a `match args.request` dispatch at the bottom (`compute` vs `interface`).

Each is invoked as:

```bash
conda run … python3 /entrypoint.py --apiUrl … --request interface|compute …
```

**Consequence:** the lightweight `interface` request (just builds a UI dict and
POSTs it; the interactive, user-facing path) pays the **full** top-level import
bill, including heavy ML libraries it never uses.

## Section 2 — Import cost

### Every-run, unavoidable (~200–300 ms total)

`numpy`, `shapely`, `annotation_client.*`, `annotation_utilities.*`, `skimage`.

From `-X importtime` on the light worker:

| Module / child | Cumulative |
|----------------|------------|
| `annotation_client.tiles` → `tifffile` | ~66 ms |
| `annotation_utilities.annotation_tools` → `matplotlib` | ~53 ms |
| `numpy` | ~57 ms |
| `girder_client` / `requests` | ~30 ms |

### Cross-cutting lazy win

`annotation_utilities/annotation_tools.py:3`:

```python
import matplotlib.colors as mcolors
```

is used exactly **once**, at line 324 (`mcolors.to_rgb(layer['color'])`).
`annotation_tools` is imported by ~40 workers, so nearly every worker pays
~53 ms for a single color conversion.

### Heavy ML imports at module top (only needed for `compute`)

These should be deferred into `compute()`:

| Worker | Location | Imports |
|--------|----------|---------|
| cellposesam | `cellposesam/entrypoint.py:12-14` | `deeptile` → `cellpose_segmentation`, `cellpose` + `torch`; interface at `:26` only lists models |
| stardist | `stardist/entrypoint.py:13` | `from stardist.models import StarDist2D` (TensorFlow) |
| sam2_propagate + all `sam2_*` / `sam_*` | `sam2_propagate/entrypoint.py:21-24` | `torch`, `sam2.*` |
| deepcell | `deepcell/entrypoint.py:11-12` | `deeptile` → `deepcell_mesmer_segmentation` |
| piscis predict / train | — | `torch`, `piscis` |
| cellpose_train | — | `cellpose` |
| blob_random_forest_classifier | — | `sklearn`, `pandas` (compute only) |

These import in **seconds**; deferring them makes the `interface` request
near-instant with no cost to `compute`.

## Section 3 — How to measure (reproducible)

```bash
PY=/root/miniconda3/envs/worker/bin/python3.12   # miniconda on arm64, miniforge on amd64; python3.11 for some workers
IMG=properties/blob_intensity:latest
```

**(a) `conda run` vs direct python:**

```bash
time docker run --rm --entrypoint conda $IMG run --no-capture-output -n worker python3 -c "pass"
time docker run --rm --entrypoint $PY   $IMG -c "pass"
```

**(b) full app startup, imports + argparse, NO network (dry/no-op mode):**

```bash
time docker run --rm $IMG --help
```

**(c) import breakdown:**

```bash
docker run --rm --entrypoint $PY $IMG -X importtime -c \
  "import annotation_client.workers, annotation_client.tiles, annotation_utilities.annotation_tools, numpy, skimage.draw" \
  2> importtime.log
sort -t'|' -k2 -n -r importtime.log | head -25
```

**Reading `-X importtime`:** columns are `self | cumulative | module` in
microseconds; indentation = nesting. Sort by column 2 (cumulative) and read the
least-indented rows = top-level modules including their children. A
multi-second low-indent row (e.g. `torch`) is the answer. Here the largest is
`annotation_client.workers` at ~127 ms cumulative — proof that imports aren't
the bottleneck.

`--help` is the no-op/dry mode: it exercises the interpreter and all imports
without `--apiUrl`/`--token`, isolating startup from network.

## Section 4 — Dockerfile / base image (size-only, NOT startup)

`workers/base_docker_images/Dockerfile.worker_base`: `FROM ubuntu:jammy`; full
Miniconda/Miniforge (env = 2.4 GB); workers layer `COPY entrypoint.py` + a
`conda run` `ENTRYPOINT`.

- **`r-base`** installed (lines 13 and 18) and **completely unused** (no `.R`
  files, no `rpy2`, no `Rscript` anywhere). `/usr/lib/R` = 63 MB + apt deps.
  Dead weight from a template.
- **`build-essential`** (`:10`), **`git`** (`:19`), `libpython3-dev`, dev headers
  left in the final image — no multi-stage prune (the `FROM base AS build` stage
  is the shipped image).
- Full conda, not slim.

**Framing:** images are local and not pulled at runtime, so **none** of these
affect startup latency — these are disk/build/registry wins only, and lower
priority for the stated goal.

## Section 5 — Shared startup boilerplate

Every worker builds a Girder client (`UPennContrastWorkerClient` /
`…PreviewClient`); `__init__` just stores `apiUrl`/`token` (no network — see
`annotation_client/annotations.py`), so construction is cheap. The Girder API
handshake is real, but it is network RTT to the server, **not** container
startup — slimming won't touch it. Profile it separately (e.g. is `interface`
making redundant round-trips?). `worker_client/worker_client/worker_client.py:1-15`
imports are all light.

## Section 6 — Ranked opportunities

### A. In-container app-startup wins (these move the needle)

| # | Opportunity | Files | Payoff | Effort | Risk / caveat |
|---|-------------|-------|--------|--------|---------------|
| **1 ⭐** | **Drop `conda run` from `ENTRYPOINT`.** Bake activation into the image: `ENV PATH=/…/envs/worker/bin:$PATH` plus the activate.d-exported vars, then `ENTRYPOINT ["python","/entrypoint.py"]`; or migrate conda bases to the micromamba pattern the test base already uses. | All 76 Dockerfiles (cleanest via the 3 base images) | **~0.8–1.0 s off EVERY job** (compute AND interface), ~70% of light-worker startup | Low–Med (mechanical, touches all workers) | **MUST preserve conda's activation env vars.** Confirmed empirically: direct python loses `GDAL_DATA`, `PROJ_DATA`, `GDAL_DRIVER_PATH` (set by `gdal-activate.sh`/`proj4-activate.sh`), which GDAL/rasterio/large_image need at runtime (CRS transforms, format drivers). A naive `--entrypoint python` switch would **silently break** image-processing/rasterio workers. Bake those vars via `ENV`, or use micromamba `_entrypoint.sh` (~0.22 s, ~0 overhead, runs activation). |
| 2 | Defer heavy ML imports into `compute()`. | cellposesam:12-14, stardist:13, sam2_*/sam_*:21-24, deepcell:11-12, piscis, cellpose_train | `interface` ~instant for ML workers (saves **seconds** of torch/TF/cellpose import on the interactive path), no cost to compute | Low per worker | Verify `interface()` truly doesn't use the lib (confirmed for those checked); add a comment so it isn't "tidied" back up. |
| 3 | Lazy/remove matplotlib in shared module. | `annotation_tools.py:3` → move into fn at `:324`, or replace `mcolors.to_rgb` with a hex/named-color parser | ~50 ms off ~40 workers | Low | None if moved as-is; replacing `to_rgb` needs a correct parser. |
| 4 | Trim other always-on imports if cheap (confirm `tifffile`/`girder_client` truly needed at top). | — | ~50–100 ms | Low | Secondary. |

### B. Image-size wins (do NOT affect startup — disk/build only)

| # | Opportunity | Files | Payoff | Effort |
|---|-------------|-------|--------|--------|
| 5 | Remove `r-base` from base apt installs (unused). | `Dockerfile.worker_base:13,18`; `Dockerfile.image_processing_worker_base:13,18` | ~63 MB + r deps | Low |
| 6 | Multi-stage build: drop `build-essential`/`git`/dev headers from the final layer. | Both conda base Dockerfiles | Hundreds of MB | Med |
| 7 | `conda clean -afy` / slimmer env after `env create`. | Base Dockerfiles | Trims the 2.4 GB env | Low |

## Single highest-leverage change

**#1 — replace `conda run` in the `ENTRYPOINT` with a pre-activated
direct-`python` entrypoint** (preserving GDAL/PROJ activation vars).

It is the only change that helps **every worker on every invocation**; it is
~70% of measured light-worker startup; and the micromamba base proves activation
can cost ~0 ms. Validate a rasterio/large_image worker end-to-end before rolling
the base-image change out to all 76 images.

## Status / Next steps

- Items **#1 (drop `conda run`)**, **#2 (defer ML imports)**, and **#3 (lazy
  matplotlib)** are **implemented and verified** (see the Implementation Log).
- The entrypoint change (#1) has been **rolled out to all production workers** —
  the 19 shared-base workers (via the base images) plus the 28 GPU /
  self-contained production Dockerfiles (see the 2026-06-27 *continued* log).
- Image-size wins **#5 (drop r-base)** and **#7 (conda clean)** are **done and
  verified**. **#6 (multi-stage prune)** and **#4 (trim other always-on imports)**
  remain open.
- The GPU-worker import changes (#2) were **build-host validated 2026-06-28**:
  all 13 GPU workers build + smoke-clean on a g4dn/driver-580 host. Three
  piscis-specific issues surfaced and were fixed there (a `run_worker.sh`
  build-context error, a vestigial jax/flax dependency, and a torch import on the
  `interface` path) — see the 2026-06-28 log. Still untested: a real end-to-end
  `compute` run against live data.

## Implementation Log — 2026-06-27

Items #1, #2, and #3 from the ranked opportunities above were implemented and
verified. The audit content above (measurements, tables, rankings) is unchanged
and remains the source of truth for the *why*; this log records the *what was
done*.

### The fix mechanism (item #1)

A new shared script **`workers/base_docker_images/run_worker.sh`** replaces the
per-worker `conda run --no-capture-output -n worker python /entrypoint.py`
`ENTRYPOINT`. It activates the conda env **in-process** (sets
`CONDA_PREFIX`/`PATH`, sources the env's `activate.d` hooks so `GDAL_DATA`,
`PROJ_DATA`, and `GDAL_DRIVER_PATH` survive) and then `exec`s the env python —
avoiding conda's second Python process. It auto-detects miniforge (amd64) vs
miniconda (arm64).

This directly addresses the headline finding (the `conda run` wrapper is ~70% of
startup) while preserving the GDAL/PROJ activation vars flagged as the key risk
in opportunity #1.

### 1. Baked into the 2 conda base images — DONE & VERIFIED

- `Dockerfile.worker_base` and `Dockerfile.image_processing_worker_base` now
  `COPY run_worker.sh` and set
  `ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]`.
- The **19 production workers** built `FROM` these bases had their per-worker
  `conda run` `ENTRYPOINT` removed (replaced with an inheritance comment) so they
  inherit the fast one.
- `worker_client` was added to `worker-base` (mirroring image-processing-base) so
  CPU workers can build `FROM` it cleanly.

**Measured results** (arm64, full startup to ready via `--help`, median of 5):

| Worker | Before | After |
|--------|--------|-------|
| blob_intensity | 1.21 s | 0.41 s |
| crop | 1.22 s | 0.44 s |
| blob_metrics | ~1.2 s | 0.40 s |
| registration | ~1.2 s | 0.51 s |

~0.8 s saved per job (~65–70%), matching the audit's predicted ~0.8–1.0 s. Verified
GDAL/PROJ env vars are **identical** to the conda-run reference; `large_image` /
`rasterio` import fine. Tests pass: blob_metrics 9, registration 20, crop 14,
blob_intensity 9.

### 1b. Refactored 3 inline production workers onto worker-base — DONE & VERIFIED

`laplacian_of_gaussian`, `line_scan_worker`, and
`blob_colony_two_color_intensity_worker` were self-contained
`FROM ubuntu:jammy` builds. They are now `FROM nimbusimage/worker-base:latest`
(so they inherit the fast entrypoint), dropping:

- their duplicated base build,
- the stale `Kitware/UPennContrast` client clone (the base uses the current
  `arjunrajlaboratory/NimbusImage`),
- the hardcoded x86_64 miniforge.

Made arch-agnostic, so their obsolete `Dockerfile_M1` files were **deleted** and
their docker-compose entries pinned to the literal `Dockerfile`.
line_scan_worker's `__main__` was aligned to the canonical `match args.request`
dispatch pattern. Verified: all three inherit `run_worker.sh`, are fast
(0.39 / 0.54 / 0.38 s), and line_scan tests (7) pass.

**Not refactored** (left on their current path, by decision):
`blob_random_forest_classifier` (needs sklearn/mahotas) and `ai_analysis`
(py3.9 + anthropic).

### 3. matplotlib lazy import (item #3) — DONE & VERIFIED

`annotation_utilities/annotation_tools.py` — moved
`import matplotlib.colors as mcolors` from the module top into the only function
that uses it (`process_and_merge_channels`). Verified matplotlib no longer loads
when `annotation_tools` is imported (~50 ms saved across ~all workers, as
predicted in opportunity #3).

### 2. Deferred heavy ML imports into compute() (item #2) — DONE, STATIC-VALIDATED ONLY

Heavy library imports were moved from module top into `compute()` (and the
helper functions that use them) for all **13 GPU production workers**, so the
`interface` request and container startup no longer pay multi-second ML imports:

- cellpose, cellpose_train, cellposesam, condensatenet, deepcell, stardist
- sam2_automatic_mask_generator, sam2_fewshot_segmentation,
  sam_fewshot_segmentation, sam2_propagate, sam2_refine, sam2_video
- piscis_predict, piscis_train

Libraries deferred: torch / tensorflow / cellpose / deeptile / stardist / sam2 /
segment_anything / deepcell / piscis. Two transitive cases were also handled:

- condensatenet's `from condensatenet import CondensateNetPipeline` moved into
  its segmentation function.
- sam_fewshot's module-level `_patch_torchvision_nms()` call moved into
  `compute()`.

A few genuinely-unused heavy imports were dropped during this pass (e.g.
sam2_propagate's `SAM2AutomaticMaskGenerator`, sam2_video's `build_sam2`).

**Verified (static):** all 13 pass `python3 -m py_compile`; independent grep
confirms NO module-level heavy imports remain **except**
`from piscis.paths import MODELS_DIR` in the two piscis workers (used by
`interface()` to list models). **[Update 2026-06-28: that exception turned out to
defeat the speedup for piscis — `from piscis.paths import …` runs
`piscis/__init__.py` → `from piscis.core import Piscis` → `import torch`, so the
`interface` request still paid the full ~4 s torch import. Fixed; see the
2026-06-28 log.]**

> **CAVEAT — GPU workers require build-host validation before deploy. [RESOLVED
> 2026-06-28 — see the 2026-06-28 log.]** The item-#2 changes were
> static-validated only (py_compile + usage analysis); GPU images cannot be built
> locally (no CUDA/torch). All 13 GPU workers have since been built on an amd64
> g4dn host (driver 580) and smoke-tested — imports resolve at runtime and the
> workers start on GPU. The piscis `MODELS_DIR` question is answered: **yes**, it
> triggered piscis's `__init__` → `import torch` (~4 s on every `interface`); now
> fixed with a torch-free `MODELS_DIR`. Still untested: a full `compute` run
> against live image data.

## Implementation Log — 2026-06-27 (continued): rollout + image-size wins

### 4. Entrypoint rollout to all remaining production workers (task a) — DONE

The base-image bake (above) covered only workers built `FROM` the 2 shared
bases. The **28 remaining production Dockerfiles** — which build their own conda
env (GPU workers on `nvidia/cuda`, plus the inline CPU workers) — were converted
to `run_worker.sh` directly (`COPY` + `chmod` + the `run_worker.sh` `ENTRYPOINT`,
replacing `conda run`), including their `_M1` variants:

- 13 GPU workers: cellpose, cellpose_train, cellposesam, condensatenet, deepcell,
  stardist, sam2_automatic_mask_generator, sam2_fewshot_segmentation,
  sam_fewshot_segmentation, sam2_propagate, sam2_refine, sam2_video,
  piscis_predict, piscis_train
- deconwolf (compose, `nvidia/cuda`)
- ai_analysis, blob_random_forest_classifier (inline CPU)

`run_worker.sh` auto-detects the self-built env (`/root/miniforge3/envs/worker`
on amd64, `/root/miniconda3/envs/worker` on arm64), so the same script works for
these workers unchanged.

Also: **sam_automatic_mask_generator** (flagged in review — it had been missed in
the item-#2 pass) had its module-level `torch` / `segment_anything` deferred into
its `segment_image` helper, and its entrypoint converted.

**Verified:** all 28 Dockerfiles statically checked (COPY present, `run_worker.sh`
ENTRYPOINT, no `conda run` remaining). Runtime-validated on
`blob_random_forest_classifier` (a self-built-env worker — env auto-detected,
`--help` exits 0 at ~0.8 s, 4 tests pass). **GPU workers still require build-host
(amd64) validation** per the caveat above.

Deliberately **left on `conda run`**: deprecated workers (not in the production
manifest) and the `$BASE_IMAGE`-ARG workers (unknown env layout).

### 5. Image-size wins #5 + #7 (task b) — DONE & VERIFIED

- Removed the unused **`r-base`** from both base images' apt installs.
- Added **`conda clean --all --yes`** after env setup in both bases.

| Base image | Before | After |
|------------|--------|-------|
| `nimbusimage/worker-base` | 5.18 GB | 4.74 GB |
| `nimbusimage/image-processing-base` | 9.65 GB | 9.09 GB |

Verified `Rscript` is gone from both bases and that workers on each base still
build and import (`blob_metrics`; `crop` with GDAL vars intact). These are
disk/build wins only — they do not affect startup latency.

### Review fixes

- The new files (`run_worker.sh`, this doc) are now tracked.
- `build_all_property_and_annotation_workers.sh` points the 3 refactored workers
  at the literal `Dockerfile` (their `$DOCKERFILE`→`_M1` files were deleted).
- piscis predict/train entrypoints normalized CRLF→LF (`git diff --check` clean).

## Implementation Log — 2026-06-28: interface-path import deferral + unused-import sweep

Grounded in fresh measurements: stdlib and `skimage` (lazy submodule loading) are
~0; the real weight is `scipy.spatial` (~0.28s), `geopandas` (~0.25s), `pandas`
(~0.20s), `numpy` (~0.10s). Crucial rule confirmed empirically: **a dead/unused
import only costs startup if its library isn't already loaded by a needed
module.** `annotation_utilities.annotation_tools` (imported by ~all workers) always
pulls `numpy`+`shapely`, so cutting a worker's direct `import numpy` saves ~0
(verified: removing all 11 of crop's dead imports → 0.44s→0.44s).

### Deferred heavy compute-only libs into compute()/helpers (interface path faster)
17 production entrypoints: libs used only in compute/helpers (never in
`interface`/`preview`) moved into those functions, each with a `# Lazy import: …`
comment. Libraries deferred: pandas, geopandas, scipy, sklearn, mahotas,
large_image, rasterio.

- connect_to_nearest / connect_sequential / connect_timelapse — pandas, geopandas, scipy
- blob_overlap_worker (geopandas); blob_random_forest_classifier (pandas, sklearn, mahotas);
  children_count_worker (pandas); line_scan_worker (pandas, scipy)
- crop / gaussian_blur / h_and_e_deconvolution / histogram_matching / rolling_ball /
  deconwolf — large_image
- stardist / piscis_train — rasterio (STATIC-VALIDATED ONLY; GPU build host needed)

Measured interface-path proxy (`--help`, median of 3): **blob_random_forest 0.82→0.32s
(−0.50s)**, crop 0.43→0.34s, connect_to_nearest 0.33s, registration 0.39s.

**Caveat — `registration`:** only `large_image` deferred; `StackReg` left at module
top because the tests `patch('entrypoint.StackReg')` and the code references
`StackReg.TRANSLATION/.RIGID_BODY/.AFFINE` class constants. Not worth a test rewrite
for the small pystackreg win.

### Unused-import sweep
Removed **143 unused module-level imports across 41 entrypoints** (AST: zero
name-references; conservative — single-line top-level imports only, kept used names
in multi-name lines, skipped conditional/parenthesized imports). Most are ~0 startup
(stdlib / `numpy`-shadowed / lazy `skimage`) — pure hygiene. The few real wins were
dead heavy imports: `imageio` (crop / histogram_matching / registration) and
`scipy.spatial.distance` (point_to_nearest_blob_distance). The sweep also covered
deprecated workers — dead `cv2`/`numpy` were removed from `line_length_worker`,
`point_to_nearest_point_distance`, and `point_to_nearest_connected_point_distance`
(these are `$BASE_IMAGE` workers that can't be built locally, so they are
py_compile/reference-validated only). `ai_analysis` was left untouched (slated for
deprecation).

### Verification
All buildable worker-profile test suites pass — connect family, crop,
blob_random_forest (4), blob_overlap, children_count (7), line_scan (7),
point_to_nearest_blob_distance (13), gaussian_blur (15), h_and_e_deconvolution (9),
histogram_matching (16), rolling_ball (17), registration (20), deconwolf (38), plus
the unchanged-logic property workers. The suspicious removals were confirmed safe
(h_and_e doesn't use skimage's `hed2rgb`; blob_point_count doesn't use the local
`point_in_polygon`; parent_child doesn't use `defaultdict`). GPU workers
(cellpose/sam2/stardist/piscis) are static-validated (py_compile + reference
analysis) only and need amd64 build-host runtime validation.

### Remaining future work (not done)
- **#6 (multi-stage prune)** — drop `build-essential` / `git` / dev headers from
  the final base layers. The riskier image-size win; warrants its own pass.
- ~~**GPU build-host validation** of the deferral + unused-import changes.~~
  **DONE 2026-06-28** — all 13 GPU workers built + smoke-clean on a g4dn host
  (see the 2026-06-28 log).
- Deprecated / `$BASE_IMAGE` workers remain on `conda run` (not shipped).

## Implementation Log — 2026-06-28: GPU build-host validation + piscis fixes

The item-#2 (deferral) and entrypoint changes were validated on an amd64 GPU
build host (`g4dn.2xlarge`, Tesla T4, driver 580 — the same AMI prod workers
run), building each worker's image and smoke-testing it (`--help` full startup +
an `--gpus all` probe that imports exactly the libs deferred into `compute()`).

**All 13 GPU workers build + smoke-clean.** The lazy-import refactor is
runtime-clean — no `NameError`/`ImportError` from deferring imports into
`compute()`; every torch worker reports `torch.cuda.is_available() == True`, and
stardist (TF 2.11) sees the GPU. (`--help` exits 0 for all, confirming
`run_worker.sh` activation + module-level imports load.)

Three piscis-specific issues — none catchable by static validation — were found
and fixed (all on `claude/worker-startup-latency`):

1. **`run_worker.sh` build-context error (commit `8748cac`).** piscis is the only
   GPU worker built via compose with `context: .` = the `workers/annotations/piscis/`
   subdir, so the new `COPY ./workers/base_docker_images/run_worker.sh` (correct
   for the 12 repo-root-context workers) didn't resolve → both piscis images
   failed to build. Fixed by building piscis from repo-root context (compose
   `context: ../../..`) and re-rooting the piscis-local `COPY`s.

2. **Vestigial jax/flax (commit `60afa80`).** A first attempt added `jax[cuda12]`
   to "GPU-enable" piscis, but the current piscis is **torch-only**: the worker
   entrypoints import only `torch`, and neither piscis 1.0.0 (PyPI) nor the
   zjniu/Piscis source declares jax/flax. jax entered solely via a leftover
   `pip install flax`. Both the `jax[cuda12]` line and `pip install flax` were
   removed; piscis runs on torch (image ~30 GB vs ~45 GB with the jax stack).

3. **`interface` request paid a ~4 s torch import (commit `bc70262`).** This is
   the caveat flagged in the item-#2 log. Both entrypoints **and** the worker's
   `utils.py` had module-level `from piscis.paths import MODELS_DIR`, and
   `from piscis.paths import …` runs `piscis/__init__.py` → `from piscis.core
   import Piscis` → `import torch`. Because the entrypoints `import utils` at
   module load (and `interface()` calls `utils.list_girder_models`), torch loaded
   on every interface request despite being deferred in `compute()`. Fixed by
   defining `MODELS_DIR = Path.home()/'.piscis'/'models'` (plain pathlib, no
   piscis import) in `utils.py` and pointing both entrypoints at
   `utils.MODELS_DIR`. **Verified on a GPU build:** loading `entrypoint.py` no
   longer imports torch (`importtime` clean); interface-path import **0.35–0.41 s
   vs ~4.2 s**; model list unchanged; compute path + GPU unaffected.

**Still untested:** a real end-to-end `compute` run against live Girder image
data — all of the above covers build + startup + import resolution + GPU
visibility, not a full inference pass. Verify piscis predict/train on the
platform opportunistically once this merges.
