FROM nimbusimage/sam2-worker-base:latest

COPY ./workers/annotations/sam2_refine/utils.py /
COPY ./workers/annotations/sam2_refine/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="SAM2 Refiner" \
      interfaceCategory="SAM2" \
      description="Uses SAM2 to refine annotations" \
      annotationShape="polygon" \
      defaultToolName="SAM2 Refiner"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
