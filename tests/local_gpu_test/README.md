# Local GPU Worker Testing

Test worker Docker containers locally without a Girder server. A lightweight Flask mock server impersonates the Girder API, serving test images and capturing worker outputs.

## Quick Start

### Option 1: Shell script (simplest)

```bash
# Install mock server dependencies (host machine, one-time)
pip install flask numpy tifffile

# Test random_squares (no GPU needed)
./tests/local_gpu_test/run_local_test.sh annotations/random_squares:latest

# Test cellposesam with GPU
./tests/local_gpu_test/run_local_test.sh annotations/cellposesam:latest --gpu

# Test a property worker
./tests/local_gpu_test/run_local_test.sh properties/blob_intensity:latest

# Use custom parameters
./tests/local_gpu_test/run_local_test.sh annotations/cellposesam:latest --gpu --params-file my_params.json
```

### Option 2: Docker Compose

```bash
# Start mock server, then run a worker
docker compose -f tests/local_gpu_test/docker-compose.gpu-test.yml run random-squares-test

# With GPU
docker compose -f tests/local_gpu_test/docker-compose.gpu-test.yml run cellposesam-test
```

### Option 3: Manual (most control)

```bash
# Terminal 1: Start mock server
python tests/mock_girder_server/server.py --images-dir tests/fixtures/images --port 5555

# Terminal 2: Run worker container
docker run --rm --network host annotations/random_squares:latest \
  --apiUrl http://localhost:5555/api/v1 \
  --token fake_test_token \
  --request compute \
  --parameters '{"datasetId":"test_dataset", ...}' \
  --datasetId test_dataset

# Terminal 3: Inspect results
curl http://localhost:5555/api/v1/_test/status
curl http://localhost:5555/api/v1/_test/annotations | python -m json.tool
```

## How It Works

```
┌──────────────────────┐         ┌─────────────────────┐
│   Worker Container   │  HTTP   │   Mock Girder Server │
│  (GPU, real ML code) │ ──────► │   (Flask, port 5555) │
│                      │         │                      │
│  entrypoint.py runs  │         │  Serves:             │
│  exactly as in prod  │         │  - Test TIFF images  │
│                      │         │  - Tile metadata     │
│                      │         │                      │
│                      │         │  Captures:           │
│                      │         │  - Annotations       │
│                      │         │  - Property values   │
└──────────────────────┘         └─────────────────────┘
```

Workers don't know they're talking to a mock. They use `--network host` (or shared Docker network) to reach the mock server at the same URL they'd normally use for Girder.

## Test Images

Generate a synthetic test image:
```bash
pip install tifffile numpy
python tests/fixtures/images/generate_test_image.py --output-dir tests/fixtures/images
```

Or place your own TIFF files in `tests/fixtures/images/`. The mock server loads the first TIFF it finds and serves it as the dataset.

Supported TIFF shapes:
- `(H, W)` — single channel
- `(C, H, W)` — multi-channel
- `(Z, C, H, W)` — multi-channel + Z stack
- `(T, Z, C, H, W)` — full 5D

## Custom Parameters

The `run_local_test.sh` script has built-in defaults for `random_squares`, `cellposesam`, and property workers. For other workers, create a JSON file:

```json
{
    "datasetId": "test_dataset",
    "type": "segmentation",
    "id": "test-tool-id",
    "name": "My Test",
    "image": "annotations/my_worker:latest",
    "channel": 0,
    "assignment": {"XY": 0, "Z": 0, "Time": 0},
    "tags": ["test"],
    "tile": {"XY": 0, "Z": 0, "Time": 0},
    "connectTo": {"layer": null, "tags": [], "channel": null},
    "workerInterface": {
        "MyParam": 42
    },
    "scales": {
        "pixelSize": {"unit": "mm", "value": 0.000325},
        "tStep": {"unit": "s", "value": 1},
        "zStep": {"unit": "m", "value": 1}
    }
}
```

Then: `./run_local_test.sh annotations/my_worker:latest --params-file params.json`

## Inspecting Results

After a worker runs, outputs are saved to `tests/fixtures/output/`:

- `annotations.json` — all annotations the worker created
- `property_values.json` — all property values computed
- `connections.json` — any connections created

You can also query the mock server while it's running:

```bash
# Status summary
curl http://localhost:5555/api/v1/_test/status

# All annotations
curl http://localhost:5555/api/v1/_test/annotations

# All property values
curl http://localhost:5555/api/v1/_test/property_values

# Reset for another run
curl -X POST http://localhost:5555/api/v1/_test/reset

# Save outputs to disk
curl -X POST http://localhost:5555/api/v1/_test/save
```

## GPU Workers

For GPU workers (cellposesam, stardist, piscis, etc.):

```bash
# Shell script
./run_local_test.sh annotations/cellposesam:latest --gpu

# Manual docker run
docker run --rm --network host --gpus all annotations/cellposesam:latest ...
```

Make sure:
- NVIDIA Container Toolkit is installed
- `nvidia-smi` works inside Docker: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
