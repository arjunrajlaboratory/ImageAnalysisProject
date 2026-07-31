# ---------------------------------------------------------------------------
# Multi-stage build (see todo/ml-worker-image-size.md).
#
# Nothing in this worker compiles a CUDA extension -- condensatenet is pure
# Python on top of PyTorch / transformers -- so the CUDA *devel* image only
# serves to give the build stage a compiler for any source-only pip dependency.
# The runtime stage ships on the much smaller CUDA *runtime* image and copies
# over just the finished conda env, the trees that are pip-installed in editable
# mode, and the baked-in model snapshots.
#
# The previous `FROM base as build` was a no-op: nothing was ever copied between
# stages, so the toolchain, the miniforge installer, the pip wheel cache and the
# conda package cache all stayed in the shipped image.
#
# PyTorch never uses the base image's CUDA libraries: it loads the cuDNN /
# cuBLAS / cuFFT / NCCL shipped inside its own `nvidia-*` pip wheels under
# site-packages. That is why the runtime stage does not need a `cudnn` tag.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS build

# pip's wheel cache is a full second copy of the torch + nvidia-* wheels
# (~4 GB) and is worthless inside an image.
ENV PIP_NO_CACHE_DIR=1

# Build-time toolchain only; none of this is carried into the runtime stage.
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

# Copy and create conda environment
COPY ./workers/annotations/condensatenet/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install NimbusImage annotation client. Shallow clone: full history is dead
# weight in an image.
RUN git clone --depth 1 https://github.com/arjunrajlaboratory/NimbusImage/ && rm -rf /NimbusImage/.git
RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/

# Install annotation_utilities
COPY ./annotation_utilities /annotation_utilities
RUN pip install /annotation_utilities

# Install DeepTile for tiling support
# The DeepTile checkout is ~140 MB but the importable package is 160 KB: the
# rest is .git, sample `data` (56 MB) and `notebooks` (37 MB). It is installed
# editable, so the tree has to survive into the runtime stage -- prune it here
# rather than ship the samples.
RUN git clone --depth 1 https://github.com/arjunrajlaboratory/DeepTile && \
    rm -rf /DeepTile/.git /DeepTile/data /DeepTile/notebooks /DeepTile/tests
RUN pip install -e /DeepTile

# Install worker_client for batch processing
COPY ./worker_client /worker_client
RUN pip install /worker_client

# Install CondensateNet package
RUN pip install git+https://github.com/arjunrajlaboratory/condensatenet.git

# Download CondensateNet models (bake into image for faster startup). These are
# real file copies (snapshot_download local_dir_use_symlinks=False), so /models
# is self-contained and is all the runtime stage needs.
RUN python -m condensatenet download --output /models/condensatenet
RUN python -m condensatenet download --repo-id rajlab/condensatenet-v1 --output /models/condensatenet-v1

# Drop conda's package cache / tarballs -- another full copy of the env.
RUN conda clean --all --yes

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

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
# subtree is needed from the NimbusImage checkout, not the whole repo. Note that
# `conda` itself is deliberately not copied -- run_worker.sh activates the env by
# path, so the base conda install is build-time only.
COPY --from=build /root/miniforge3/envs/worker /root/miniforge3/envs/worker
COPY --from=build /NimbusImage/devops/girder/annotation_client /NimbusImage/devops/girder/annotation_client
COPY --from=build /DeepTile /DeepTile
# entrypoint.py hardcodes /models/condensatenet and /models/condensatenet-v1.
COPY --from=build /models /models

# Copy entrypoint
COPY ./workers/annotations/condensatenet/entrypoint.py /

WORKDIR /

LABEL isUPennContrastWorker="" \
      isGPUWorker="true" \
      isAnnotationWorker="" \
      interfaceName="CondensateNet segmentation" \
      interfaceCategory="CondensateNet" \
      description="Segments biomolecular condensates using CondensateNet deep learning model" \
      annotationShape="polygon" \
      defaultToolName="Condensates"

# Fast env activation in place of `conda run` (see todo/worker-startup-latency.md)
COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]
