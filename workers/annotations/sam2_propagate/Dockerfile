FROM nimbusimage/sam2-worker-base:latest

COPY ./workers/annotations/sam2_propagate/utils.py /
COPY ./workers/annotations/sam2_propagate/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="SAM2 propagator" \
      interfaceCategory="SAM2" \
      description="Uses SAM2 to propagate annotations through time or Z-slices" \
      annotationShape="polygon" \
      defaultToolName="SAM2 propagator"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
