# ---------------------------------------------------------------------------
# Multi-stage build (see todo/ml-worker-image-size.md).
#
# `build` runs on the CUDA *devel* image because sam2's setup.py compiles a CUDA
# extension (sam2/_C) that needs nvcc.
# Nothing at runtime needs nvcc, the CUDA headers or the static libs, so the
# final stage ships on the much smaller CUDA *runtime* image and copies over
# only the finished conda env plus the trees that are pip-installed in
# editable mode.
#
# PyTorch never uses the base image's CUDA libraries: it loads the cuDNN /
# cuBLAS / cuFFT / NCCL shipped inside its own `nvidia-*` pip wheels under
# site-packages. The devel image's copy of all of that was pure duplication.
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

COPY ./workers/annotations/sam2_fewshot_segmentation/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

RUN pip install rtree shapely

# Shallow clones: full history is dead weight in an image.
RUN git clone --depth 1 https://github.com/arjunrajlaboratory/NimbusImage/ && rm -rf /NimbusImage/.git

RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/

RUN mkdir -p /code
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /code/sam2 && rm -rf /code/sam2/.git
# Compiles the optional sam2._C CUDA extension (csrc/connected_components.cu)
# here, where nvcc exists. It is the only reason this stage needs `devel`.
RUN pip install -e /code/sam2

# Change directory to the checkpoints directory
WORKDIR /code/sam2/checkpoints
# Download the checkpoints into the checkpoints directory
RUN ./download_ckpts.sh
# Change back to the root directory
WORKDIR /

COPY ./annotation_utilities /annotation_utilities
RUN pip install /annotation_utilities

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
# subtree is needed from the NimbusImage/UPennContrast checkout, not the whole
# repo. Note that `conda` itself is deliberately not copied -- run_worker.sh
# activates the env by path, so the base conda install is build-time only.
COPY --from=build /root/miniforge3/envs/worker /root/miniforge3/envs/worker
COPY --from=build /NimbusImage/devops/girder/annotation_client /NimbusImage/devops/girder/annotation_client
COPY --from=build /code/sam2 /code/sam2
COPY ./workers/annotations/sam2_fewshot_segmentation/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isGPUWorker="true" \
      isAnnotationWorker="" \
      interfaceName="SAM2 few-shot segmentation" \
      interfaceCategory="SAM2" \
      description="Uses SAM2 features for few-shot segmentation based on training annotations" \
      annotationShape="polygon" \
      defaultToolName="SAM2 few-shot"

# Fast env activation in place of `conda run` (see todo/worker-startup-latency.md)
COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]
