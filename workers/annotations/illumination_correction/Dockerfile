FROM nimbusimage/image-processing-base:latest

# Keep libvips' own worker pool serialized. The converter still streams tiles
# from the multi-source document and never materializes the mosaic in RAM.
ENV VIPS_CONCURRENCY=1

SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]
RUN pip install "nd2==0.11.3" && \
    pip uninstall --yes pyvips && \
    pip install --no-cache-dir --no-deps \
      --find-links https://girder.github.io/large_image_wheels \
      "pyvips==3.1.1.8.18.2" && \
    conda clean --all --yes && \
    python -c "import pyvips; assert pyvips.API_mode; assert tuple(pyvips.version(i) for i in range(3)) == (8, 18, 2)"

COPY ./workers/annotations/illumination_correction/entrypoint.py /
COPY ./workers/annotations/illumination_correction/refinement.py /
COPY ./workers/annotations/illumination_correction/illumination.py /
COPY ./workers/annotations/illumination_correction/pipeline.py /

LABEL isUPennContrastWorker="" \
      isGPUWorker="false" \
      isAnnotationWorker="" \
      workerVersion="1.0.3" \
      interfaceName="Stitch Refinement + Illumination Correction" \
      interfaceCategory="Image Processing" \
      description="Refines composite translations and corrects raw-tile illumination; existing annotations are not shifted" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Stitch Refinement + Illumination Correction"

# ENTRYPOINT inherited from image-processing-base (run_worker.sh /entrypoint.py).
