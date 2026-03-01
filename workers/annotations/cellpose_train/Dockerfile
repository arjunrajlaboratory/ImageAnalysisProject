FROM nimbusimage/cuda-ml-worker-base:latest

SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install cellpose
RUN pip install cellpose==3.1.1.2

# Download models
COPY ./workers/annotations/cellpose_train/download_models.py /
RUN python /download_models.py

COPY ./workers/annotations/cellpose_train/utils.py /
COPY ./workers/annotations/cellpose_train/girder_utils.py /
COPY ./workers/annotations/cellpose_train/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Cellpose retrain" \
      interfaceCategory="Cellpose" \
      description="Retrains Cellpose models based on user-selected images" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Cellpose retrain" \
      annotationShape="polygon"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
