# ---------------------------------------------------------------------------
# Multi-stage build (see todo/ml-worker-image-size.md).
#
# Nothing in this worker compiles a CUDA extension -- cellpose is pure Python on
# top of PyTorch -- so the CUDA *devel* image only serves to give the build stage
# a compiler for any source-only pip dependency. The runtime stage ships on the
# much smaller CUDA *runtime* image and copies over just the finished conda env,
# the trees that are pip-installed in editable mode, and the model cache.
#
# The previous `FROM base as build` was a no-op: nothing was ever copied between
# stages, so the toolchain, the miniforge installer, the pip wheel cache and the
# conda package cache all stayed in the shipped image.
#
# PyTorch never uses the base image's CUDA libraries: it loads the cuDNN /
# cuBLAS / cuFFT / NCCL shipped inside its own `nvidia-*` pip wheels under
# site-packages. That is why the runtime stage does not need a `cudnn` tag.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS build

# pip's wheel cache is a full second copy of the torch + nvidia-* wheels
# (~4 GB) and is worthless inside an image.
ENV PIP_NO_CACHE_DIR=1

# Build-time toolchain only; none of this is carried into the runtime stage.
# `r-base` (listed twice) and `software-properties-common` were never used.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qy tzdata && \
  apt-get install -qy \
  build-essential \
  wget \
  git \
  libffi-dev \
  libssl-dev \
  libjpeg-dev \
  zlib1g-dev \
  libpython3-dev && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniforge3/bin:$PATH"
ARG PATH="/root/miniforge3/bin:$PATH"

RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh

COPY ./workers/annotations/cellposesam_train/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

RUN pip install rtree shapely

# Shallow clone: full history is dead weight in an image.
RUN git clone --depth 1 https://github.com/Kitware/UPennContrast/ && rm -rf /UPennContrast/.git

RUN pip install -r /UPennContrast/devops/girder/annotation_client/requirements.txt
RUN pip install -e /UPennContrast/devops/girder/annotation_client/

# models_config.py is the single source of truth for the built-in checkpoints,
# so it must be present before download_models.py runs. It is also imported by
# entrypoint.py, so the runtime stage copies it in again.
COPY ./workers/annotations/cellposesam_train/models_config.py /
COPY ./workers/annotations/cellposesam_train/download_models.py /
RUN python /download_models.py

COPY ./annotation_utilities /annotation_utilities
RUN pip install /annotation_utilities

COPY ./worker_client /worker_client
RUN pip install /worker_client

# Drop conda's package cache / tarballs -- another full copy of the env.
RUN conda clean --all --yes

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL isUPennContrastWorker=True
LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qy \
  tzdata \
  ca-certificates && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniforge3/envs/worker/bin:$PATH"

# The finished conda env, plus every tree that was pip-installed with `-e` and
# therefore has to stay at its build-time path. Only the annotation_client
# subtree is needed from the UPennContrast checkout, not the whole repo. Note
# that `conda` itself is deliberately not copied -- run_worker.sh activates the
# env by path, so the base conda install is build-time only.
COPY --from=build /root/miniforge3/envs/worker /root/miniforge3/envs/worker
COPY --from=build /UPennContrast/devops/girder/annotation_client /UPennContrast/devops/girder/annotation_client
# cellpose 4.x resolves the base checkpoints from ~/.cellpose/models
# (models.MODEL_DIR). The fine-tuned models this worker produces go to its own
# /root/.cellposesam/models, which entrypoint.py creates at runtime.
COPY --from=build /root/.cellpose /root/.cellpose

COPY ./workers/annotations/cellposesam_train/models_config.py /
COPY ./workers/annotations/cellposesam_train/utils.py /
COPY ./workers/annotations/cellposesam_train/girder_utils.py /
COPY ./workers/annotations/cellposesam_train/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isGPUWorker="true" \
      isAnnotationWorker="" \
      interfaceName="Cellpose-SAM retrain" \
      interfaceCategory="Cellpose" \
      description="Fine-tunes Cellpose-SAM models based on user-selected images" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Cellpose-SAM retrain" \
      annotationShape="polygon"

# Fast env activation in place of `conda run` (see todo/worker-startup-latency.md)
COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]
