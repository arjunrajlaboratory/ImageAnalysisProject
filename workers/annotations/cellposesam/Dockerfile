FROM nimbusimage/cuda-ml-worker-base:latest

SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install DeepTile for tiling support
RUN git clone https://github.com/arjunrajlaboratory/DeepTile
RUN pip install -e /DeepTile

# Install cellpose
RUN pip install cellpose==4.0.1 deeptile

# Download models
COPY ./workers/annotations/cellposesam/download_models.py /
RUN python /download_models.py

COPY ./workers/annotations/cellposesam/utils.py /
COPY ./workers/annotations/cellposesam/girder_utils.py /
COPY ./workers/annotations/cellposesam/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isAnnotationWorker=True \
      interfaceName="Cellpose-SAM" \
      interfaceCategory="Cellpose" \
      annotationShape="polygon" \
      description="Uses Cellpose-SAM to find cells and nuclei" \
      defaultToolName="Cellpose-SAM"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
