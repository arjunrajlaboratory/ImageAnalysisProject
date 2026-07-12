FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base
LABEL isUPennContrastWorker=True
LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qy tzdata && \
  apt-get install -qy software-properties-common python3-software-properties && \
  apt-get update && apt-get install -qy \
  build-essential \
  wget \
  python3 \
  libffi-dev \
  libssl-dev \
  libjpeg-dev \
  zlib1g-dev \
  git \
  libpython3-dev && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniforge3/bin:$PATH"
ARG PATH="/root/miniforge3/bin:$PATH"

RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh

FROM base AS build

# Create conda environment (numpy/scipy/scikit-image/tifffile/large_image system deps)
COPY ./workers/annotations/image_restoration/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install NimbusImage annotation client
RUN git clone https://github.com/arjunrajlaboratory/NimbusImage/
RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/

# Install annotation_utilities
COPY ./annotation_utilities /annotation_utilities
RUN pip install /annotation_utilities

# Install large_image for image processing (zarr source supports multi-dimensional images)
RUN pip install large-image-source-zarr large-image-source-tiff large-image-converter large-image --find-links https://girder.github.io/large_image_wheels

# PyTorch (CUDA 12.1 wheels), then the restoration-method dependencies. torch is
# intentionally installed via pip (not conda) to match the CUDA base image's
# runtime exactly.
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install careamics "cellpose>=3,<4" triton tifffile
# FluoResFM inference deps (see entrypoint.py restore_fluoresfm + the vendored
# repo's 3_0_test_it2i.py / 3_1_test_i2i.py imports):
#   open_clip_torch + transformers -> BiomedCLIP text embedder (unet_sd_c backbone)
#   huggingface_hub                -> optional embedder download (1_1_download_embedder.py)
#   torchinfo                      -> imported by models.care / models.dfcan / models.unifmir
#   timm                           -> imported by models.unifmir
# flash-attn is intentionally NOT installed: the vendored CrossAttention falls
# back to plain attention when flash_attn is absent, and only uses it for
# self-attention (never for the text cross-attention this worker relies on).
RUN pip install open_clip_torch transformers huggingface_hub torchinfo timm

# --- ZS-DeconvNet (zero-shot denoise + deconvolution) ---
# Vendored for provenance/reference and potential future use of its PSF/data
# tooling. entrypoint.py's restore_zs_deconvnet() implements a self-contained
# reimplementation of the published zero-shot algorithm rather than driving
# this repo's shell-script CLI directly -- see IMAGE_RESTORATION.md for why.
# Pinned to tag v1.0 (commit 04d2c21), the only tagged release as of this
# writing.
RUN git clone https://github.com/TristaZeng/ZS-DeconvNet.git /zs_deconvnet && \
    cd /zs_deconvnet && git checkout 04d2c21

# --- FluoResFM (pretrained, text-conditioned foundation model) ---
# Vendored from the canonical author repo (Qiqi Lu; see also the
# napari-fluoresfm plugin at https://github.com/qiqi-lu/napari-fluoresfm and
# the related UNiFMIR work at https://github.com/cxm12/UNiFMIR). Pinned to
# tag v1.0.1. Weights are NOT bundled here (see download_models.py) -- they
# are distributed via Google Drive/Baidu Yun and must be supplied at runtime
# via the FLUORESFM_WEIGHTS environment variable / volume mount.
#
# Real inference API (see entrypoint.py restore_fluoresfm):
#   * text-conditioned FluoResFM = models.unet_sd_c.UNetModel conditioned on a
#     BiomedCLIP text embedding (models.biomedclip_embedder.BiomedCLIPTextEmbedder),
#     per 3_0_test_it2i.py. The checkpoint + a small example model come from the
#     repo README's download links (Google Drive / Baidu Yun).
#   * The BiomedCLIP embedder weights (open_clip_config.json +
#     open_clip_pytorch_model.bin) come from HuggingFace
#     'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
#     (repo's 1_1_download_embedder.py) and are supplied at runtime via
#     FLUORESFM_EMBEDDER_DIR (not baked in -- the .bin is ~0.4 GB).
RUN git clone https://github.com/qiqi-lu/fluoresfm.git /fluoresfm && \
    cd /fluoresfm && git checkout v1.0.1
ENV FLUORESFM_REPO_PATH=/fluoresfm
# Default runtime weights location; override by mounting a checkpoint here or
# by overriding FLUORESFM_WEIGHTS entirely. Left unset/missing by default --
# restore_fluoresfm() sendError()s with instructions if no file exists here.
ENV FLUORESFM_WEIGHTS=/opt/fluoresfm_weights/fluoresfm.pt
# Directory holding the BiomedCLIP text-embedder files (open_clip_config.json +
# open_clip_pytorch_model.bin) for the text-conditioned 'unet_sd_c' backbone.
# Not populated at build time; mount/download the two files here at runtime.
ENV FLUORESFM_EMBEDDER_DIR=/opt/fluoresfm_weights/biomedclip

# models_config.py is the single source of truth for the built-in Cellpose3
# restoration checkpoints, so it must be present before download_models.py runs.
COPY ./workers/annotations/image_restoration/models_config.py /
COPY ./workers/annotations/image_restoration/download_models.py /
# FLUORESFM_WEIGHTS_URL is an optional build ARG for internal mirrors; if
# unset, the FluoResFM weight download is skipped without failing the build.
ARG FLUORESFM_WEIGHTS_URL=""
ENV FLUORESFM_WEIGHTS_URL=${FLUORESFM_WEIGHTS_URL}
RUN python /download_models.py

# Copy entrypoint
COPY ./workers/annotations/image_restoration/entrypoint.py /

WORKDIR /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Image Restoration" \
      interfaceCategory="Image Processing" \
      description="Restores images (Noise2Void/N2V2, Cellpose3, ZS-DeconvNet, FluoResFM)" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Image Restoration"

# Fast env activation in place of `conda run` (see todo/worker-startup-latency.md)
COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]
