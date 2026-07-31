# TODO Registry

Master index of deferred work, technical debt, and future improvements for the ImageAnalysisProject repository.

## How to use this registry

- Each TODO has a unique ID, a detailed file in `todo/`, and a tracking row below.
- When completing a TODO, update its status here and add a resolution note to the detailed file.
- When discovering new deferred work, create a new `todo/<slug>.md` file and add a row here.

## Active TODOs

| ID | Title | Status | Priority | File | Related PR |
|----|-------|--------|----------|------|------------|
| TODO-001 | ML worker build optimization (mamba + shared base images) | Deferred | Medium | [ml-worker-build-optimization.md](ml-worker-build-optimization.md) | [#132](https://github.com/arjunrajlaboratory/ImageAnalysisProject/pull/132) |
| TODO-002 | Worker startup latency audit & slimming plan (drop `conda run`) | In progress (#1–#3 done & verified; GPU workers static-validated, need build-host validation) | High | [worker-startup-latency.md](worker-startup-latency.md) | — |
| TODO-003 | SAM/SAM2 worker image size (35–40 GB → est. 10–12 GB) | In progress (Dockerfiles rewritten; **not yet built or measured**) | Medium | [ml-worker-image-size.md](ml-worker-image-size.md) | — |

## Completed TODOs

| ID | Title | Completed | Resolution |
|----|-------|-----------|------------|
| — | — | — | — |
