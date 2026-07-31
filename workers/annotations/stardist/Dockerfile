# ---------------------------------------------------------------------------
# Multi-stage build (see todo/ml-worker-image-size.md).
#
# Nothing in this worker compiles a CUDA extension -- stardist ships manylinux
# wheels -- so the CUDA *devel* image only serves to give the build stage a
# compiler for any source-only pip dependency. The runtime stage ships on the
# CUDA *runtime* image and copies over just the finished conda env, the trees
# that are pip-installed in editable mode, and the model cache.
#
# The previous `FROM base as build` was a no-op: nothing was ever copied between
# stages, so the toolchain, the miniforge installer, the pip wheel cache and the
# conda package cache all stayed in the shipped image.
#
# NOTE the `cudnn8` tag on the runtime stage, which the torch-based workers do
# not need. TensorFlow 2.11 predates the `tensorflow[and-cuda]` extra (2.14), so
# its PyPI wheel declares no `nvidia-*` dependencies and dlopens libcudnn.so.8 /
# libcublas / libcufft from the *image*. Dropping to a plain `-runtime` tag would
# silently lose the GPU (TF logs "Could not load dynamic library libcudnn.so.8"
# and falls back to CPU) rather than fail the build.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS build

# pip's wheel cache is a full second copy of the tensorflow wheel and is
# worthless inside an image.
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

COPY ./workers/annotations/stardist/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Shallow clones: full history is dead weight in an image. The DeepTile checkout
# is ~140 MB but the importable package is 160 KB -- the rest is .git, sample
# `data` (56 MB) and `notebooks` (37 MB). It is installed editable, so the tree
# has to survive into the runtime stage; prune it here rather than ship it.
RUN git clone --depth 1 https://github.com/arjunrajlaboratory/DeepTile && \
    rm -rf /DeepTile/.git /DeepTile/data /DeepTile/notebooks /DeepTile/tests
RUN pip install -e /DeepTile
RUN pip install rtree shapely

RUN git clone --depth 1 https://github.com/Kitware/UPennContrast/ && rm -rf /UPennContrast/.git

RUN pip install -r /UPennContrast/devops/girder/annotation_client/requirements.txt
RUN pip install -e /UPennContrast/devops/girder/annotation_client/

# Bakes the pretrained models into the keras cache so the first run does not
# have to download them. That directory is copied into the runtime stage.
COPY ./workers/annotations/stardist/download_models.py /
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
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

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
COPY --from=build /DeepTile /DeepTile
# csbdeep's from_pretrained() goes through keras.utils.get_file, which caches
# under ~/.keras/models -- that is where download_models.py left the two
# pretrained StarDist2D models.
COPY --from=build /root/.keras /root/.keras

COPY ./workers/annotations/stardist/utils.py /
COPY ./workers/annotations/stardist/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isGPUWorker="true" \
      isAnnotationWorker=True \
      interfaceName="Stardist" \
      interfaceCategory="Stardist" \
      annotationShape="polygon" \
      description="Uses Stardist to find cells and nuclei" \
      defaultToolName="Stardist"

# Fast env activation in place of `conda run` (see todo/worker-startup-latency.md)
COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]
