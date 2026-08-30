FROM nimbusimage/image-processing-base:latest

# BaSiCPy 2.0 uses a PyTorch backend. Install the cross-platform CPU build from
# conda-forge, then install BaSiCPy without its stale scipy<1.13 metadata pin;
# the worker base intentionally carries a newer SciPy/NumPy stack.
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]
RUN conda install --yes -c conda-forge \
      "pytorch>=2,<3" \
      "pydantic>=2,<3" \
      pooch \
      tqdm && \
    pip install "torch-dct==0.1.6" && \
    pip install --no-deps "basicpy==2.0.0" && \
    python -c "import dask.array; from basicpy import BaSiC; print(BaSiC.__name__)" && \
    conda clean --all --yes

COPY ./workers/annotations/illumination_correction/entrypoint.py /
COPY ./workers/annotations/illumination_correction/illumination.py /

LABEL isUPennContrastWorker="" \
      isGPUWorker="false" \
      isAnnotationWorker="" \
      interfaceName="Illumination Correction" \
      interfaceCategory="Image Processing" \
      description="Corrects uneven illumination in stitched microscopy images using validated channel-specific models" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Illumination Correction"

# ENTRYPOINT inherited from image-processing-base (run_worker.sh /entrypoint.py).
