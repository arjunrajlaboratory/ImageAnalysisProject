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
| TODO-003 | ML worker image size (multi-stage builds across the GPU fleet) | In progress (wave 1 SAM/SAM2 merged; wave 2 cellpose ×4 / stardist / condensatenet / piscis ×2 rewritten but **not yet built or measured**; `deconwolf` still to do) | Medium | [ml-worker-image-size.md](ml-worker-image-size.md) | [#160](https://github.com/arjunrajlaboratory/ImageAnalysisProject/pull/160) |
| TODO-004 | `channelCheckboxes` list-shaped values: identify the submitter, repair saved configs | Open (workers hardened and now reject the shape; front end surveyed, submitter unidentified) | Medium | [channelcheckboxes-serialization.md](channelcheckboxes-serialization.md) | [#162](https://github.com/arjunrajlaboratory/ImageAnalysisProject/pull/162) |
| TODO-005 | Make scoped worker builds fail when the Compose service is absent | Open | Medium | [scoped-worker-build-validation.md](scoped-worker-build-validation.md) | — |

## Completed TODOs

| ID | Title | Completed | Resolution |
|----|-------|-----------|------------|
| — | — | — | — |
