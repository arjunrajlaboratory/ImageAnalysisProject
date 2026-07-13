---
name: nimbus-worker-scaffold
description: Scaffold a brand-new NimbusImage Docker worker end-to-end in this repository. Use whenever the user wants to add, create, or port a NEW worker — a segmentation/annotation worker, a property-computation worker, an image-processing worker, or a test worker — or says things like "add a worker for X", "wrap this model/tool as a NimbusImage worker", "create a new annotation/property worker", or "scaffold a worker". Covers the full set of files a new worker needs (entrypoint.py, Dockerfile plus an optional Dockerfile_M1, tests, WORKERNAME.md, docker-compose registration, REGISTRY.md) and the Docker label set — including the mandatory isGPUWorker label that governs GPU/CPU queue routing. Scope boundary: this is the create-from-scratch workflow only. For FIXING, debugging, or hardening an EXISTING worker (crashes, deploy errors, applying a fix across sibling workers) use nimbus-worker-hardening; for the runtime Girder/large_image API details (image access, annotation CRUD, coordinate conventions) it defers to the nimbus-interface skill.
---

# Scaffolding a NimbusImage Worker

Adding a worker is the single most common change in this repo, and it is easy to
ship one that *runs* but is missing a label, a test, or a registry entry — gaps
that only surface in a production deploy. This skill exists so a new worker
lands complete and correct the first time.

A worker is a Docker image that NimbusImage runs as a **fresh container per job**
(`docker run --rm <image> --apiUrl ... --token ... --request compute|interface
--parameters ...`). It pulls images/annotations from a Girder backend and returns
either annotations or property values.

**This skill covers the *structure* of a worker. For the runtime API** (image
access, annotation CRUD, coordinate conventions, local testing), read the
`nimbus-interface` skill — don't re-derive that here.

## Step 1 — Decide the worker's shape

Three questions determine every downstream choice. Settle them first (ask the
user if the request is ambiguous):

1. **What does it output?**
   - Creates annotations (segmentation, spot detection, generators) →
     **annotation worker** (`workers/annotations/<name>/`). Label
     `isAnnotationWorker` + `annotationShape` (`polygon` | `point` | `line`).
   - Computes numbers on existing annotations → **property worker**
     (`workers/properties/{blobs,points,lines,connections}/<name>_worker/`).
     Label `isPropertyWorker` + `annotationShape`.
   - Processes images and writes a new image back to Girder (blur, registration,
     restoration) → annotation-style worker that uploads a TIFF, not annotations.

2. **Does it need a GPU?** ML models on CUDA → yes. Classical CV / numpy → no.
   This sets the base image *and* the mandatory `isGPUWorker` label (Step 3).

3. **What's the interface?** Which of the interface types (`number`, `text`,
   `select`, `checkbox`, `channel`, `channelCheckboxes`, `tags`, `layer`,
   `notes`) does the user need, and what does each return? The type→return-type
   table lives in the project `CLAUDE.md`; the `tags` type returns a **plain
   list of strings** (a common crash source), and `channelCheckboxes` returns a
   `dict`. `workers/annotations/sample_interface/entrypoint.py` demonstrates
   every type.

## Step 2 — Write `entrypoint.py`

Every entrypoint has the same three parts. Copy a working template from
`references/templates.md` rather than writing from scratch — pick the one that
matches Step 1:

- **`interface(image, apiUrl, token)`** — builds the UI dict and calls
  `client.setWorkerImageInterface(image, interface)`. Give every field a
  `displayOrder`; add a `notes` field at `displayOrder: 0` describing the tool.
- **`compute(datasetId, apiUrl, token, params)`** — the work.
  - Annotation workers should use `WorkerClient` (`worker.process(f, f_annotation=...)`),
    which handles Batch XY/Z/Time iteration and upload for you. Model on
    `workers/annotations/random_squares/entrypoint.py`.
  - Property workers build `{annotation_id: {prop: value}}`, wrap as
    `{datasetId: property_value_dict}`, and call
    `add_multiple_annotation_property_values`. Model on
    `workers/properties/blobs/blob_intensity_worker/entrypoint.py`.
- **`__main__`** — the argparse block with the `match args.request` dispatch to
  `compute` / `interface`. It is boilerplate; copy it verbatim.

Read the field values out of `params['workerInterface']` using the return types
from the table above. Validate required inputs (especially `tags`) early and
`sendError` with an actionable message — a worker that crashes with a bare
traceback gives the user nothing to act on.

## Step 3 — Write the Dockerfile(s) and the label set

Pick the base image from Step 1:

| Worker kind | Base image | ENTRYPOINT |
|---|---|---|
| Property / simple annotation (conda) | `nimbusimage/worker-base:latest` | inherited from base (omit ENTRYPOINT) |
| Image-processing (needs extra CV libs) | `nimbusimage/image-processing-base:latest` | inherited from base |
| GPU / ML | `nvidia/cuda:<ver>-cudnn*-devel-ubuntu22.04` + conda (multi-stage) | `run_worker.sh /entrypoint.py` |
| Test/demo worker | `nimbusimage/test-worker-base:latest` (micromamba) | `_entrypoint.sh python /entrypoint.py`, `COPY --chown=$MAMBA_USER` |

**The label block is not optional and not cosmetic — the dispatcher reads it.**
Every production `Dockerfile` (and any `Dockerfile_M1`, in the *final* stage of a
multi-stage build) must set:

```dockerfile
LABEL isUPennContrastWorker="" \
      isGPUWorker="true"  # or "false" — MANDATORY, no default \
      isAnnotationWorker="" \        # OR isPropertyWorker="" for property workers
      annotationShape="polygon" \    # polygon | point | line (property + shaped annotation workers)
      interfaceName="Human-readable name" \
      interfaceCategory="Grouping shown in the UI" \
      description="One line describing what it does"
```

`isGPUWorker` decides whether the job goes to the `gpu` or `cpu` Celery queue.
**An unlabeled image silently falls back to the `gpu` queue in production** and
logs a warning — so a CPU worker missing the label wastes a GPU slot, and this
is exactly the gap PRs #148/#153 existed to close across every worker. Never
leave it unset.

A **`Dockerfile_M1`** is *optional*. It's the CPU/arm64 variant that
`build_workers.sh` selects when `MAC_DEVELOPMENT_MODE=true` (or on an arm64
host) — the two are the same mechanism, not alternatives — and it's the only way
to build the worker on a Mac. Add one only if you want Mac buildability; plenty
of GPU workers (including `cellposesam`) ship without a `Dockerfile_M1` and
simply can't be built on a Mac. Note this is separate from a *runtime* CPU
fallback (try CUDA, fall back to CPU), which lets the production x86 image run on
a GPU-less host. See `references/templates.md` for the multi-stage CUDA +
label-in-final-stage pattern.

## Step 4 — Tests

Create `tests/` with `__init__.py`, `test_<name>.py`, and `Dockerfile_Test`.
Tests run against a mocked `annotation_client` — no live server. Cover at
minimum: (a) `interface()` produces the expected fields/types, and (b)
`compute()` produces the expected annotations/properties given mocked tiles.
Mock `UPennContrastWorkerPreviewClient`, `tiles.UPennContrastDataset`, and
`annotations.UPennContrastAnnotationClient`. Template in
`references/templates.md`; model on
`workers/annotations/random_squares/tests/`.

Per the project's test-first convention, write these tests **before** the
implementation when practical, and make sure they fail before they pass.

## Step 5 — Register and document

A worker Claude forgot to register won't build; one it forgot to document
drifts. Do all of these:

1. **`docker-compose.yml`** — add a build service and a `_test` service. Match
   the `profiles` to the worker kind:
   - annotation/property production worker → `profiles: ["worker", "annotations"]`
     or `["worker", "properties"]`; test service → `profiles: ["test"]`.
   - test/demo worker → `profiles: ["testworker"]`; its test → `["testworkertest"]`.

   ```yaml
     my_worker:
       build:
         context: .
         dockerfile: ./workers/annotations/my_worker/Dockerfile
       image: annotations/my_worker:latest
       depends_on: [worker-base]
       profiles: ["worker", "annotations"]
     my_worker_test:
       build:
         context: .
         dockerfile: ./workers/annotations/my_worker/tests/Dockerfile_Test
       image: annotations/my_worker:test
       depends_on: [my_worker]
       profiles: ["test"]
   ```

2. **`WORKERNAME.md`** — a hand-written doc in the worker directory (uppercase,
   e.g. `MY_WORKER.md`). Cover: title/description, how it works, an interface
   parameter table, computed properties (property workers), implementation
   notes, and GPU/limitations. Model on `random_squares/RANDOM_SQUARES.md`
   (concise) or `deconwolf/DECONWOLF.md` (comprehensive). Keep the labels in the
   Dockerfile and the doc consistent.

3. **`REGISTRY.md`** — add the worker to the correct category table. This is the
   master index and must be updated on every add/remove/rename.

Do **not** run `generate_worker_docs.py --force` — it overwrites hand-written
docs with stubs. The auto-doc hooks are disabled on purpose; docs are manual.

## Step 6 — Build and verify

```bash
./build_workers.sh <name>                          # build just this worker
./build_workers.sh --build-and-run-tests <name>    # build + run its tests
# ML/GPU worker:
./build_machine_learning_workers.sh
# test/demo worker:
./build_test_workers.sh --build-and-run-tests <name>
# Mac dev of a GPU worker (builds Dockerfile_M1 — the worker must have one):
MAC_DEVELOPMENT_MODE=true ./build_workers.sh <name>
```

Don't claim the worker is done until the build succeeds and its tests pass —
show the output.

## Completion checklist

Before calling a new worker complete, confirm every item — a missing one is the
usual cause of a worker that builds locally but misbehaves in deploy:

- [ ] `entrypoint.py` with `interface()`, `compute()`, and the `__main__` dispatch
- [ ] `Dockerfile` with the full label set, **including `isGPUWorker`**
- [ ] `Dockerfile_M1` **only if** the worker should be Mac-buildable (optional; it's the variant `MAC_DEVELOPMENT_MODE=true` selects)
- [ ] `tests/` with `__init__.py`, `test_<name>.py`, `Dockerfile_Test`
- [ ] `docker-compose.yml`: worker service **and** `_test` service, correct profiles
- [ ] `WORKERNAME.md` in the worker directory
- [ ] `REGISTRY.md` updated
- [ ] `./build_workers.sh --build-and-run-tests <name>` passes
