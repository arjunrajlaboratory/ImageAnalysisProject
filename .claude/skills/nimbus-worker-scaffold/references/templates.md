# Worker Templates

Copy-paste starting points for a new worker. Pick the section that matches the
worker's shape (decided in SKILL.md Step 1), then adapt. These are distilled
from working workers in the repo — when in doubt, open the cited reference
worker and follow it.

## Table of contents
- [Annotation worker (WorkerClient batch)](#annotation-worker-workerclient-batch)
- [Property worker](#property-worker)
- [The `__main__` block (all workers)](#the-__main__-block-all-workers)
- [Dockerfile — conda worker-base (CPU)](#dockerfile--conda-worker-base-cpu)
- [Dockerfile — GPU / CUDA (multi-stage)](#dockerfile--gpu--cuda-multi-stage)
- [Dockerfile — test/demo worker (micromamba)](#dockerfile--testdemo-worker-micromamba)
- [Test file + Dockerfile_Test](#test-file--dockerfile_test)

---

## Annotation worker (WorkerClient batch)

Reference: `workers/annotations/random_squares/entrypoint.py`. `WorkerClient`
handles Batch XY/Z/Time iteration and annotation upload; your function returns
polygon coordinate lists (or points/lines) for one image. Numeric batch inputs
are 1-indexed, `all` expands to every dataset coordinate, and empty fields use
the current tile.

```python
import argparse
import json
import sys

import annotation_client.workers as workers
import annotation_utilities.batch_argument_parser as batch_argument_parser
from worker_client import WorkerClient


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    interface = {
        'My Worker': {
            'type': 'notes',
            'value': 'What this worker does, in one or two sentences.',
            'displayOrder': 0,
        },
        'Channel': {
            'type': 'channel',
            'required': True,
            'tooltip': 'Channel to process.',
            'displayOrder': 1,
        },
        'Some number': {
            'type': 'number', 'min': 1, 'max': 100, 'default': 10,
            'unit': 'pixels', 'displayOrder': 2,
        },
        # The three Batch fields come from the shared helper -- never hand-write
        # them. They accept 1-indexed ranges or `all`, numbered from display_order.
        **batch_argument_parser.batch_interface_fields(display_order=3),
    }
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    worker = WorkerClient(datasetId, apiUrl, token, params)
    channel = params['workerInterface']['Channel']
    some_number = float(worker.workerInterface['Some number'])

    def process_image(image):
        # image is a (H, W) numpy array for the requested frame.
        # Return a list of polygons, each a list of (x, y) tuples in image space.
        polygons = []
        # ... detect objects, build polygons ...
        return polygons

    worker.process(process_image, f_annotation='polygon',
                   stack_channels=[channel],
                   progress_text='Processing')
```

For image-processing workers that upload a TIFF instead of annotations, follow
`workers/annotations/histogram_matching/entrypoint.py` (the `large_image` sink
pattern documented in `CLAUDE.md`).

---

## Property worker

Reference: `workers/properties/blobs/blob_intensity_worker/entrypoint.py`.

```python
import argparse
import json
import sys

import annotation_client.workers as workers
import annotation_client.tiles as tiles
from annotation_client.utils import sendError, sendWarning
import annotation_utilities.annotation_tools as annotation_tools


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    interface = {
        'My Property': {'type': 'notes', 'value': 'What this measures.', 'displayOrder': 0},
        'Channel': {'type': 'channel', 'required': True, 'displayOrder': 1},
    }
    client.setWorkerImageInterface(image, interface)


def compute(datasetId, apiUrl, token, params):
    workerClient = workers.UPennContrastWorkerClient(datasetId, apiUrl, token, params)
    datasetClient = tiles.UPennContrastDataset(apiUrl=apiUrl, token=token, datasetId=datasetId)

    channel = params['workerInterface']['Channel']

    annotationList = workerClient.get_annotation_list_by_shape('polygon', limit=0)
    annotationList = annotation_tools.get_annotations_with_tags(
        annotationList,
        params.get('tags', {}).get('tags', []),
        params.get('tags', {}).get('exclusive', False),
    )

    property_value_dict = {}
    for annotation in annotationList:
        # ... compute values ...
        property_value_dict[annotation['_id']] = {'MyValue': float(value)}

    workerClient.add_multiple_annotation_property_values({datasetId: property_value_dict})
```

Note the two different `tags` shapes: a `tags` **interface field** returns a
plain list, while `params['tags']` for property filtering is
`{'tags': [...], 'exclusive': bool}`. See `CLAUDE.md` and the `nimbus-interface`
skill.

---

## The `__main__` block (all workers)

Identical across workers — copy verbatim.

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='My worker')
    parser.add_argument('--datasetId', type=str, required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str, required=True, action='store')
    args = parser.parse_args(sys.argv[1:])

    params = json.loads(args.parameters)
    match args.request:
        case 'compute':
            compute(args.datasetId, args.apiUrl, args.token, params)
        case 'interface':
            interface(params['image'], args.apiUrl, args.token)
```

---

## Dockerfile — conda worker-base (CPU)

For property workers and simple annotation workers. ENTRYPOINT is inherited
from the base (do not redefine it).

```dockerfile
FROM nimbusimage/worker-base:latest

COPY ./workers/properties/blobs/my_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isGPUWorker="false" \
      isPropertyWorker="" \
      annotationShape="polygon" \
      interfaceName="My property" \
      interfaceCategory="Intensity" \
      description="Compute my property for blobs"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py)
```

Use `nimbusimage/image-processing-base:latest` instead if the worker needs the
heavier image-processing environment.

---

## Dockerfile — GPU / CUDA (multi-stage)

Reference: `workers/annotations/cellposesam/Dockerfile`. Note the label block
lives in the **final** stage, `isGPUWorker="true"`, and the fast-activation
`run_worker.sh` entrypoint.

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base
LABEL isUPennContrastWorker=True
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# ... install miniforge, system deps ...

FROM base as build
COPY ./workers/annotations/my_worker/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install the NimbusImage annotation client required by every worker entrypoint.
RUN git clone --depth 1 https://github.com/arjunrajlaboratory/NimbusImage /NimbusImage
RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt && \
    pip install -e /NimbusImage/devops/girder/annotation_client/

# ... model-specific pip installs and downloads at build time ...
COPY ./workers/annotations/my_worker/entrypoint.py /
COPY ./annotation_utilities /annotation_utilities
RUN pip install /annotation_utilities
COPY ./worker_client /worker_client
RUN pip install /worker_client

LABEL isUPennContrastWorker=True \
      isGPUWorker="true" \
      isAnnotationWorker=True \
      interfaceName="My GPU worker" \
      interfaceCategory="Segmentation" \
      annotationShape="polygon" \
      description="Uses <model> to segment cells"

COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]
```

A `Dockerfile_M1` (CPU/arm64) is optional: it's the variant `build_workers.sh`
selects under `MAC_DEVELOPMENT_MODE=true` (or arm64), and the only way to build
this worker on a Mac. Many GPU workers skip it and aren't Mac-buildable. That is
separate from a *runtime* CPU fallback (try CUDA, fall back to CPU) — a
common addition so the production x86 image still runs on a GPU-less host. Pin
transitive build dependencies that break under new resolvers (see the
`nimbus-worker-hardening` skill on the `setuptools`/`pkg_resources` class of
build breakage).

---

## Dockerfile — test/demo worker (micromamba)

Reference: `workers/annotations/random_squares/Dockerfile`. Runs as
`$MAMBA_USER`; copy with `--chown`.

```dockerfile
FROM nimbusimage/test-worker-base:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER ./workers/annotations/my_test_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isGPUWorker="false" \
      isAnnotationWorker="" \
      interfaceName="My test worker" \
      interfaceCategory="Testing" \
      description="Generates test annotations"

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/entrypoint.py"]
```

---

## Test file + Dockerfile_Test

Reference: `workers/annotations/random_squares/tests/`.

`tests/test_my_worker.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from entrypoint import compute, interface


def test_interface():
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_client:
        interface('test_image', 'http://test-api', 'test-token')
        mock_client.return_value.setWorkerImageInterface.assert_called_once()
        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]
        assert 'Channel' in interface_data
        assert interface_data['Channel']['type'] == 'channel'


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute(mock_annotation_client, mock_dataset_client):
    mock_dataset_client.return_value.tiles = {
        'tileWidth': 1000, 'tileHeight': 1000,
        'IndexRange': {'IndexXY': 1, 'IndexZ': 1, 'IndexT': 1, 'IndexC': 1},
    }
    mock_dataset_client.return_value.coordinatesToFrameIndex.return_value = 0
    mock_dataset_client.return_value.getRegion.return_value = MagicMock(
        squeeze=MagicMock(return_value=np.zeros((1000, 1000))))
    params = {
        'assignment': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0, 'connectTo': {'tags': []}, 'tags': ['test'],
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'workerInterface': {'Channel': 0},
    }
    compute('test_dataset', 'http://test-api', 'test-token', params)
    mock_annotation_client.return_value.createMultipleAnnotations.assert_called_once()
```

Include an edge-case test for a dataset with **no `IndexRange`** (single-frame) —
that omission is a recurring production crash; see `nimbus-worker-hardening`.
For WorkerClient batch fields, also exercise `all` against a mocked
multi-position `IndexRange` and assert every expected coordinate is processed.

`tests/Dockerfile_Test` (conda worker) — mirror the house convention exactly
(`SHELL` activates the env, then a plain `pip install`; `python3` in the
entrypoint):

```dockerfile
FROM annotations/my_worker:latest AS test

# Install test dependencies
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]
RUN pip install pytest pytest-mock

# Copy test files
RUN mkdir -p /tests
COPY ./workers/annotations/my_worker/tests/*.py /tests
WORKDIR /tests

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python3", "-m", "pytest", "-v"]
```

`tests/Dockerfile_Test` (micromamba test worker):

```dockerfile
FROM annotations/my_test_worker:latest AS test
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install pytest pytest-mock
USER root
RUN mkdir -p /tests && chown $MAMBA_USER:$MAMBA_USER /tests
USER $MAMBA_USER
COPY ./workers/annotations/my_test_worker/tests/*.py /tests
WORKDIR /tests
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python3", "-m", "pytest", "-v"]
```

`tests/__init__.py` can be empty.
