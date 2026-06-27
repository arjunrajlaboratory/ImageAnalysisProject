# TODO-002: Worker Startup Latency Audit & Slimming Plan

**Status:** In progress
**Priority:** High
**Related PR:** —

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

- Item **#1 (drop `conda run`)** has been **implemented and verified** (see the
  Implementation Log below). Items **#2 (defer ML imports)** and **#3 (lazy
  matplotlib)** are also done. Item #4 (trim other always-on imports) and the
  Section B image-size items (#5–#7) remain open.
- The GPU-worker portion of #2 is **static-validated only** and requires
  build-host (amd64) validation before deploy — see the caveat in the log.

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
`from piscis.paths import MODELS_DIR` in the two piscis workers (legitimately
used by `interface()` to list models).

> **CAVEAT — GPU workers require build-host validation before deploy.** The
> item-#2 changes are **static-validated only** (py_compile + usage analysis);
> GPU images cannot be built locally (no CUDA/torch). Each GPU image must be
> built on the **amd64 build host** and smoke-tested: confirm imports resolve at
> runtime and a `compute` run works. Also verify whether piscis's `MODELS_DIR`
> still triggers piscis's package `__init__` at startup — if that `__init__` is
> heavy, consider deferring the `interface` model-list too.

### Remaining future work (not done)

- Rolling the entrypoint change (#1) out to the **other** GPU / self-contained
  workers' Dockerfiles also needs them to adopt `run_worker.sh` — the bake only
  covers workers built `FROM` the 2 shared bases.
- Item #4 (trim other always-on imports) remains open.
- Section B image-size wins (#5 r-base removal, #6 multi-stage, #7 conda clean)
  remain open.
