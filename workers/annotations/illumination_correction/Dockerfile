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

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Illumination Correction" \
      interfaceCategory="Image Processing" \
      description="Corrects illumination/shading/striping (BaSiC, CIDRE, CellProfiler, flat-field, destripe)" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Illumination Correction"

# ENTRYPOINT inherited from image-processing-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)
