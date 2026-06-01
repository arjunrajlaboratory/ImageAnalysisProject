# TODO-002: Pull-first / build-fallback for worker images from ECR

**Status:** Deferred
**Priority:** Low
**Related PR:** [#141 — push_workers_to_ecr.sh (produce side)](https://github.com/arjunrajlaboratory/ImageAnalysisProject/pull/141)
**Companion (consume side):** AWSDeploy PR #54 — Celery worker EC2 instances pull prebuilt worker images from ECR at startup instead of rebuilding from source.

## Summary

The "worker images on ECR" workflow has two halves:

- **Produce** (this repo, PR #141, done): `scripts/push_workers_to_ecr.sh` builds each worker for `linux/amd64` and pushes it to `nimbus/<category>/<worker>` in ECR. It honors the shared `$ECR_REGISTRY` env-var contract (commit `48d13a3`).
- **Consume** (AWSDeploy PR #54): Celery GPU workers get a read-only ECR IAM role and `docker login` to ECR at boot, then their cloud-init clones this repo and runs the build scripts.

The deferred piece is the link between them: make the **build scripts pull-first** so that when a prebuilt image exists in ECR, cloud-init pulls it instead of rebuilding from source (the whole point of the consume side). Until this lands, AWSDeploy PR #54 is a harmless no-op — workers still rebuild.

## What's done

- `push_workers_to_ecr.sh` resolves the registry from `$ECR_REGISTRY` (default: derived from the authenticated account + region). No new flag; no behavior change when unset. The full env-var contract is documented in AWSDeploy's `doc/Build_and_Push_Worker_Images_to_ECR.md` (do **not** duplicate that doc here).

## What's deferred (the work)

In `build_workers.sh` and `build_machine_learning_workers.sh` (the scripts AWSDeploy's Celery cloud-init clones and invokes), implement per-worker pull-first / build-fallback:

- If `$ECR_REGISTRY` is set and `docker pull "$ECR_REGISTRY/nimbus/<category>/<worker>:latest"` succeeds, `docker tag` it to the local tag Girder's worker discovery expects, and **skip** the build for that worker.
- Otherwise fall through to the existing build. **Behavior must be unchanged when `$ECR_REGISTRY` is unset** (strict additive change).
- Treat ML/GPU workers identically to standard workers — no separate code path.
- Reuse the existing source of truth for the worker list + category mapping rather than re-enumerating workers in a second place.

## Blockers / open questions (resolve before writing this)

These are why it was deferred rather than done as a quick additive change:

1. **What local image tag does Girder/Celery worker discovery actually expect?**
   This is the `docker tag` target. Getting it wrong silently breaks worker
   discovery, so it must be confirmed against the running system / how workers
   are registered — not guessed.

2. **The names don't line up across the scripts.** There is no single existing
   source of truth that spans all four naming spaces:

   | Thing | Example |
   |---|---|
   | Worker **directory** (what `push_workers_to_ecr.sh` enumerates) | `cellpose`, `blob_metrics_worker` |
   | **ECR repo** the push script creates | `nimbus/annotations/cellpose`, `nimbus/properties/blob_metrics_worker` |
   | Local tag the **ML build script** produces | `annotations/cellpose_worker:latest` (note the `_worker` suffix) |
   | `build_workers.sh` **compose service** name | `blob_metrics` (≠ dir `blob_metrics_worker`), `connect_time_lapse` (≠ dir `connect_timelapse`) |

   A correct pull-first needs an explicit mapping between these.

3. **`build_workers.sh` builds everything in one `docker compose build`**, not
   per-worker, so per-worker pull-first requires restructuring it into a
   per-service loop (or a pre-pass that pulls+tags available images, then tells
   compose to skip those). `build_machine_learning_workers.sh` is already
   per-worker (`docker build ... -t ...`), so the fallback is easier there.

## Verification plan (when implemented)

- `bash -n` each modified script.
- `./scripts/push_workers_to_ecr.sh --dry-run --all` still renders the expected registry URLs.
- With `$ECR_REGISTRY` **unset**, a build script still rebuilds from source (no regression).
- With `$ECR_REGISTRY` set and a worker image present in ECR, that worker is pulled + tagged, not rebuilt.

## Notes

- No new shell/Python dependencies — AWS CLI + Docker are already required.
- Keep it targeted/additive; don't refactor unrelated parts of the build scripts.
- The originating spec (from the AWSDeploy agent) wanted a single PR titled e.g.
  "Honor $ECR_REGISTRY; pull prebuilt worker images when available" covering
  both the `$ECR_REGISTRY` change (done, `48d13a3`) and this pull-first work.
