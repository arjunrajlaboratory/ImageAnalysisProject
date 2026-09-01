FROM nimbusimage/image-processing-base:latest

# Keep libvips' own worker pool serialized while large_image streams frames.
ENV VIPS_CONCURRENCY=1

# BaSiCPy 2.0 uses a PyTorch backend. Install the cross-platform CPU build and
# its runtime optimizer explicitly, then install BaSiCPy without its stale
# scipy<1.13 metadata pin. The worker base intentionally carries a newer
# SciPy/NumPy stack; the worker tests exercise real fits with darkfield off/on.
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]
RUN conda install --yes -c conda-forge \
      "pytorch>=2,<3" \
      "pydantic>=2,<3" \
      pooch \
      tqdm && \
    pip install "torch-dct==0.1.6" "hyperactive>=4.4,<5" && \
    pip install --no-deps "basicpy==2.0.0" && \
    pip uninstall --yes pyvips && \
    pip install --no-cache-dir --no-deps \
      --find-links https://girder.github.io/large_image_wheels \
      "pyvips==3.1.1.8.18.2" && \
    python -c "import dask.array; from basicpy import BaSiC; print(BaSiC.__name__)" && \
    python -c "import pyvips; assert pyvips.API_mode; assert tuple(pyvips.version(i) for i in range(3)) == (8, 18, 2)" && \
    conda clean --all --yes

COPY ./workers/annotations/stitched_illumination_correction/entrypoint.py /
COPY ./workers/annotations/stitched_illumination_correction/illumination.py /

LABEL isUPennContrastWorker="" \
      isGPUWorker="false" \
      isAnnotationWorker="" \
      workerVersion="1.0.3" \
      interfaceName="Stitched TIFF Illumination Correction" \
      interfaceCategory="Image Processing" \
      description="Stitched-TIFF fallback for grid-locked illumination correction with held-out model selection" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Stitched TIFF Illumination Correction"

# ENTRYPOINT inherited from image-processing-base (run_worker.sh /entrypoint.py).
