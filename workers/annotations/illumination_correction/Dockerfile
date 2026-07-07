FROM nimbusimage/image-processing-base:latest

# The base image's "worker" conda env already has numpy>=2.0 / scipy /
# scikit-image and large_image; activate it to install this worker's extra
# dependencies.
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# large_image wheels (mirrors deconwolf's pattern; already present in the base
# image, pinned again here for explicitness/self-containment).
RUN pip install large-image-source-zarr large-image-source-tiff large-image-converter large-image --find-links https://girder.github.io/large_image_wheels

# CPU-only torch/torchvision. The base is a plain (non-CUDA) image, so pull the
# CPU wheels explicitly -- the default PyPI torch wheel bundles a multi-GB CUDA
# runtime that would never be used here. Installing torch first also lets us
# satisfy basicpy's/SSCOR's torch dependency with the CPU build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# BaSiCPy (BaSiC shading/darkfield correction) for the `basic` method.
# Installed with --no-deps, then its runtime deps by hand: BaSiCPy 2.0.0
# hard-pins scipy<1.13, but the base env is numpy>=2.0 (which needs scipy>=1.13,
# already installed and linked by the conda scikit-image/large_image binaries).
# Letting pip honor basicpy's pin would downgrade scipy and break those
# numpy-2-ABI C extensions at runtime, so we keep the base scipy and install
# only basicpy's *other* runtime deps (torch is the CPU build installed above).
RUN pip install --no-deps basicpy && \
    pip install "hyperactive>=4.4.0,<5" torch-dct pooch "pydantic>=2.0.0" dask tqdm

# pystripe (wavelet-FFT destriping) for the `destripe` method.
RUN pip install pystripe

# --- SSCOR (deep-learning stripe self-correction) ---
# Vendored from https://github.com/lxxcontinue/SSCOR, pinned to commit
# 985479cd79bcf1359e3d9ba44bacd5f372eb2e60. It's a pytorch-CycleGAN-and-pix2pix
# -style codebase with no clean importable inference API, so `correct_sscor` in
# entrypoint.py shells out to its CLI scripts (restore.py / train.py / sampling;
# see SSCOR_REPO_PATH).
RUN git clone https://github.com/lxxcontinue/SSCOR.git /sscor && \
    cd /sscor && git checkout 985479cd79bcf1359e3d9ba44bacd5f372eb2e60

# Extra deps SSCOR's pix2pix-style pipeline needs (torch/torchvision are the CPU
# builds installed above).
RUN pip install dominate visdom opencv-python matplotlib wandb

# NOTE: SSCOR weights are NOT baked into this image -- the upstream repo
# distributes trained generator checkpoints via Google Drive, not a stable
# direct-download URL, and producing one requires an image-specific offline
# self-training stage (see ILLUMINATION_CORRECTION.md). For SSCOR mode
# "pretrained", supply a checkpoint at runtime by mounting it into the container
# and setting the SSCOR_WEIGHTS environment variable to its path (staged
# internally as the `latest_net_G_A.pth` the CycleGAN loader expects); the
# `sscor` Method sendError()s with instructions if it is unset/missing. SSCOR
# mode "self-train" needs no checkpoint (it trains one per frame).

# entrypoint.py is copied last so routine edits don't invalidate the (slow)
# dependency/clone layers above.
COPY ./workers/annotations/illumination_correction/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Illumination Correction" \
      interfaceCategory="Image Processing" \
      description="Corrects illumination/shading/striping (BaSiC, CIDRE, CellProfiler, flat-field, destripe, SSCOR)" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Illumination Correction"

# ENTRYPOINT inherited from image-processing-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)
