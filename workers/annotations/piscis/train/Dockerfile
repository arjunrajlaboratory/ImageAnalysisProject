# ---------------------------------------------------------------------------
# Multi-stage build (see todo/ml-worker-image-size.md).
#
# Nothing in this worker compiles a CUDA extension -- piscis is pure Python on
# top of PyTorch -- so the CUDA *devel* image only serves to give the build stage
# a compiler for any source-only pip dependency. The runtime stage ships on the
# much smaller CUDA *runtime* image and copies over just the finished conda env,
# the trees that are pip-installed in editable mode, and the model cache.
#
# The previous `FROM base as build` was a no-op: nothing was ever copied between
# stages, so the toolchain, the conda installer, the pip wheel cache and the
# conda package cache all stayed in the shipped image.
#
# PyTorch never uses the base image's CUDA libraries: it loads the cuDNN /
# cuBLAS / cuFFT / NCCL shipped inside its own `nvidia-*` pip wheels under
# site-packages.
#
# Miniconda was also swapped for Miniforge here, matching every other worker.
# That drops the `defaults` channel and with it the `conda tos accept` dance;
# this worker's environment.yml is only `python=3.11` + `pip`, all of which
# conda-forge provides.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

# pip's wheel cache is a full second copy of the torch + nvidia-* wheels
# (~4 GB) and is worthless inside an image.
ENV PIP_NO_CACHE_DIR=1

# Build-time toolchain, plus the opencv runtime libs (see below) -- only the
# latter are repeated in the runtime stage.
#
# `r-base` (listed twice) and `software-properties-common` were never used
# *directly*, but r-base-core's dependency tree was silently supplying the
# shared libraries `opencv-python` needs: libglib2.0-0, libx11-6, libsm6 and
# libice6 (via libxt6) and libxext6 (via libtk8.6). piscis depends on
# `opencv-python` -- not `-headless` -- and imports cv2 at module scope in
# piscis/transforms.py, so `import piscis` needs them here as well: the
# `python /download_models.py` step below imports the package. libgl1 is listed
# explicitly rather than left to whatever the CUDA base happens to carry.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qy tzdata && \
  apt-get install -qy \
  build-essential \
  wget \
  git \
  libffi-dev \
  libssl-dev \
  libjpeg-dev \
  zlib1g-dev \
  libpython3-dev \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libice6 \
  libx11-6 \
  libxext6 && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniforge3/bin:$PATH"
ARG PATH="/root/miniforge3/bin:$PATH"

RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh

COPY ./workers/annotations/piscis/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Shallow clones: full history is dead weight in an image.
RUN git clone --depth 1 https://github.com/arjunrajlaboratory/NimbusImage && rm -rf /NimbusImage/.git

RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/

# Installed non-editable, so the clone itself is build-stage only.
RUN git clone --depth 1 https://github.com/zjniu/Piscis.git && rm -rf /Piscis/.git
WORKDIR /Piscis
RUN pip install .

WORKDIR /
RUN pip install rasterio shapely

# Install the local annotation_utilities / worker_client, not a fresh clone of
# ImageAnalysisProject from GitHub: cloning built the image against whatever was
# on the default branch instead of the tree being built.
COPY ./annotation_utilities /annotation_utilities
RUN pip install /annotation_utilities

COPY ./worker_client /worker_client
RUN pip install /worker_client

# Bakes the pretrained models into ~/.piscis/models so the first run does not
# have to download them. That directory is copied into the runtime stage.
COPY ./workers/annotations/piscis/download_models.py /
RUN python /download_models.py

# Drop conda's package cache / tarballs -- another full copy of the env.
RUN conda clean --all --yes

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL isUPennContrastWorker=True
LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# libgl1 .. libxext6 are what `opencv-python` links against; piscis imports cv2
# at module scope, so the worker cannot start without them. See the build stage.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qy \
  tzdata \
  ca-certificates \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libice6 \
  libx11-6 \
  libxext6 && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniforge3/envs/worker/bin:$PATH"

# The finished conda env, plus every tree that was pip-installed with `-e` and
# therefore has to stay at its build-time path. Only the annotation_client
# subtree is needed from the NimbusImage checkout, not the whole repo. Note that
# `conda` itself is deliberately not copied -- run_worker.sh activates the env by
# path, so the base conda install is build-time only.
COPY --from=build /root/miniforge3/envs/worker /root/miniforge3/envs/worker
COPY --from=build /NimbusImage/devops/girder/annotation_client /NimbusImage/devops/girder/annotation_client
# piscis.paths.MODELS_DIR is ~/.piscis/models; this worker also writes the models
# it trains there before uploading them to Girder.
COPY --from=build /root/.piscis /root/.piscis

COPY ./workers/annotations/piscis/utils.py /
COPY ./workers/annotations/piscis/train/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isGPUWorker="true" \
      isAnnotationWorker=True \
      interfaceName="Piscis spot training" \
      interfaceCategory="Piscis" \
      description="Retrain Piscis to detect your particular spots" \
      defaultToolName="Piscis training"

# Fast env activation in place of `conda run` (see todo/worker-startup-latency.md)
COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]
