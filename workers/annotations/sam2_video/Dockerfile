FROM nimbusimage/sam2-worker-base:latest

SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Replace the standard SAM2 with the nimbus video predictor fork
RUN pip uninstall -y SAM-2
RUN git clone -b nimbus-video-predictor https://github.com/arjunrajlaboratory/segment-anything-2-nimbus.git /code/segment-anything-2-nimbus
RUN pip install -e /code/segment-anything-2-nimbus

# Download checkpoints from the nimbus fork
WORKDIR /code/segment-anything-2-nimbus/checkpoints
RUN ./download_ckpts.sh
WORKDIR /

COPY ./workers/annotations/sam2_video/utils.py /
COPY ./workers/annotations/sam2_video/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="SAM2 video" \
      interfaceCategory="SAM2" \
      description="Uses SAM2 to track video through time or Z" \
      annotationShape="polygon" \
      defaultToolName="SAM2 video"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
