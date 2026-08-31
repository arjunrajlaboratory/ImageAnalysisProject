---
name: nimbus-run-worker
description: "Run a built NimbusImage Docker worker end to end against a real local Girder-backed dataset, including resolving dataset-view URLs, minting and revoking a short-lived token, constructing compute parameters, attaching to the Nimbus Docker network, and verifying API and frontend results. Use when asked to run, execute, smoke-test, reproduce, or validate a worker on local NimbusImage data. Use nimbus-worker-scaffold to create workers, nimbus-worker-hardening to fix failures, and nimbus-interface for worker API implementation details."
---

# Running a NimbusImage Worker Locally

Run the production Docker image directly while using the live local Girder API
for all reads and writes. This exercises the image entrypoint, `compute()` code,
source downloads, result uploads, and metadata handling. It bypasses only
Celery dispatch and therefore does **not** prove CPU/GPU queue routing.

A compute run can create annotations, property values, or image items. Run it
only when the user authorized a live result; use `--dry-run` for inspection.

## Standard workflow

### 1. Check the stack and image

Run from the `ImageAnalysisProject` root. Confirm the Girder-backed Nimbus stack
and the worker image exist:

```bash
curl -fsS http://localhost:8080/api/v1/system/version
docker image inspect <worker-image>:latest
docker network inspect nimbusimage_default
```

If Girder returns HTTP 500 beyond `/system/version`, confirm MongoDB and the
rest of the Nimbus stack are running. Read `nimbus-interface` for infrastructure
diagnostics.

Inspect the image labels before a live run. Direct Docker execution ignores the
queue label, but production dispatch does not:

```bash
docker image inspect --format \
  '{{ index .Config.Labels "interfaceName" }} GPU={{ index .Config.Labels "isGPUWorker" }}' \
  <worker-image>:latest
```

Pass `--gpus all` to the bundled runner only when the image and host require it.

### 2. Resolve the dataset correctly

Workers expect the **Girder dataset folder ID** in `--datasetId`. A Nimbus URL
such as `#/datasetView/<view-id>/view` contains a dataset-view ID, not that
folder ID. Resolve it with authenticated `GET /dataset_view/<view-id>` and use
the response's `datasetId` value.

The bundled runner accepts either `--dataset-id` or `--dataset-view-url` and
performs this resolution safely. Never pass the large-image item ID merely
because `/item/<id>/tiles` works; workers use the folder to select the active
large image and access sibling resources.

### 3. Build the exact compute payload

Read the worker's `interface()`, worker documentation, and compute tests. Do not
guess field types or assume interface defaults will fill omitted saved values.
Construct the complete `params` object the worker expects, including top-level
fields such as `tile`, `tags`, `channel`, or `scales` when that worker uses them.

Write the payload to a private temporary JSON file. Keep credentials out of it:

```json
{
  "workerInterface": {
    "Example option": true,
    "Output filename": "validated-output.tiff"
  }
}
```

For a destructive or expensive worker, use a unique output name and inspect
existing results first. Never delete a prior item just to make a rerun easier
unless the user explicitly requested replacement.

### 4. Authenticate with a short-lived token and run

Keep `NIMBUS_API_KEY` in the repository's ignored `.env` or the process
environment. Never print it, enable shell tracing, or pass the long-lived API
key to `--token`.

Use the bundled runner for the standard local stack:

```bash
python3 .agents/skills/nimbus-run-worker/scripts/run_worker.py \
  --image <worker-image>:latest \
  --dataset-view-url 'http://localhost:5173/#/datasetView/<view-id>/view' \
  --parameters-file /tmp/worker-parameters.json
```

The runner:

1. reads `NIMBUS_API_KEY` without echoing it;
2. mints a temporary Girder token with `POST /api_key/token`;
3. resolves and validates the dataset folder;
4. snapshots the folder's current items;
5. runs the image with `--rm` on `nimbusimage_default`;
6. reports newly created folder items; and
7. revokes the token with `DELETE /token/session`, including on failure.

Preview the command shape without API or Docker calls:

```bash
python3 .agents/skills/nimbus-run-worker/scripts/run_worker.py \
  --image <worker-image>:latest \
  --dataset-id <dataset-folder-id> \
  --parameters-file /tmp/worker-parameters.json \
  --dry-run
```

Check authentication, dataset-view resolution, the folder, image, and Docker
network without starting the worker:

```bash
python3 .agents/skills/nimbus-run-worker/scripts/run_worker.py \
  --image <worker-image>:latest \
  --dataset-view-url 'http://localhost:5173/#/datasetView/<view-id>/view' \
  --parameters-file /tmp/worker-parameters.json \
  --preflight-only
```

For a differently named Compose project, override `--docker-network`. Override
both API URLs together when ports or service names differ:

- `--host-api-url`: reachable from the host, used for auth and verification.
- `--container-api-url`: reachable from the worker container, passed as
  `--apiUrl` (normally `http://girder:8080/api/v1`).

The current worker contract necessarily exposes the short-lived token in the
local container process arguments while it runs. Limit this to a trusted local
machine and rely on immediate revocation; never substitute the API key.

### 5. Verify the result at three boundaries

Do not treat exit code zero as sufficient.

1. **Worker report** — retain the final progress/output and check worker-specific
   metrics, warnings, counts, and quality gates.
2. **Girder state** — inspect the new or modified resources through the API.
   For an image output, check `GET /item/<id>`, wait for
   `GET /item/<id>/tiles`, confirm axes/frame count/levels, and fetch one or more
   exact `GET /item/<id>/tiles/region` crops. For annotations or properties,
   compare relevant counts and inspect representative records.
3. **Nimbus frontend** — load the dataset from a fresh page, select a new image
   in the large-image dropdown when applicable, and verify that it renders.

Compare the same coordinates and display style for before/after image crops.
For registration or restoration workers, inspect the known worst-case region,
not only an easy area.

Report the image tag/ID, branch or commit, exact non-secret parameters, output
resource IDs, quantitative gates, and visual check. State explicitly that a
direct run bypassed Celery. If queue routing itself matters, perform a second
run through Nimbus's normal UI/job path and inspect the selected queue.

## Manual command contract

If the helper cannot cover a specialized container option, preserve the same
contract and token lifecycle:

```bash
docker run --rm \
  --network nimbusimage_default \
  <worker-image>:latest \
  --datasetId <dataset-folder-id> \
  --apiUrl http://girder:8080/api/v1 \
  --token <short-lived-girder-token> \
  --request compute \
  --parameters '<complete-params-json>'
```

Use argument arrays or carefully quoted JSON; never use `eval`. Arrange token
revocation in a `finally` block or shell trap before launching the container.
