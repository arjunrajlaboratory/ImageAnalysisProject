FROM nimbusimage/cuda-ml-worker-base:latest

SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install DeepTile for tiling support
RUN git clone https://github.com/arjunrajlaboratory/DeepTile
RUN pip install -e /DeepTile

# Install cellpose
RUN pip install cellpose==3.1.1.2 deeptile

# Download models
COPY ./workers/annotations/cellpose/download_models.py /
RUN python /download_models.py

COPY ./workers/annotations/cellpose/utils.py /
COPY ./workers/annotations/cellpose/girder_utils.py /
COPY ./workers/annotations/cellpose/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isAnnotationWorker=True \
      interfaceName="Cellpose" \
      interfaceCategory="Cellpose" \
      annotationShape="polygon" \
      description="Uses Cellpose to find cells and nuclei" \
      defaultToolName="Cellpose"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
