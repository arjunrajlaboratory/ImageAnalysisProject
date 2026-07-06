FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/illumination_correction/entrypoint.py /

# The base image's "worker" conda env already has numpy/scipy/scikit-image and
# large_image; activate it to install this worker's extra dependencies.
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# large_image wheels (mirrors deconwolf's pattern; already present in the base
# image, pinned again here for explicitness/self-containment).
RUN pip install large-image-source-zarr large-image-source-tiff large-image-converter large-image --find-links https://girder.github.io/large_image_wheels

# Illumination-correction-specific Python dependencies not present in the base image.
# basicpy (BaSiCPy 2.x, PyTorch backend -- CPU-only here) and pystripe (wavelet-FFT destriping).
RUN pip install basicpy pystripe

# --- SSCOR (deep-learning stripe self-correction) ---
# Vendored from https://github.com/lxxcontinue/SSCOR, pinned to commit
# 985479cd79bcf1359e3d9ba44bacd5f372eb2e60. It's a pytorch-CycleGAN-and-pix2pix
# -style codebase with no clean importable inference API, so `correct_sscor` in
# entrypoint.py shells out to its CLI script `restore.py` (see SSCOR_REPO_PATH).
RUN git clone https://github.com/lxxcontinue/SSCOR.git /sscor && \
    cd /sscor && git checkout 985479cd79bcf1359e3d9ba44bacd5f372eb2e60

# torch itself is already present (pulled in by basicpy above); these are the
# additional deps restore.py's pix2pix-style pipeline needs. Do NOT pin torch
# here -- reuse whatever version basicpy installed.
RUN pip install torchvision dominate visdom opencv-python matplotlib wandb

# NOTE: SSCOR weights are NOT baked into this image -- the upstream repo
# distributes trained generator checkpoints via Google Drive, not a stable
# direct-download URL, and producing one requires an image-specific offline
# self-training stage (see ILLUMINATION_CORRECTION.md). Supply a checkpoint at
# runtime by mounting it into the container and setting the SSCOR_WEIGHTS
# environment variable to its path (a `latest_net_G.pth` file); the `sscor`
# Method sendError()s with instructions if it is unset/missing.

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
