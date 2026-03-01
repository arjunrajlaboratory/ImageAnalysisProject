FROM nimbusimage/sam2-worker-base:latest

COPY ./workers/annotations/sam2_automatic_mask_generator/utils.py /
COPY ./workers/annotations/sam2_automatic_mask_generator/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="SAM2 automatic mask generator" \
      interfaceCategory="SAM2" \
      description="Uses SAM2 to find all masks in the image" \
      annotationShape="polygon" \
      defaultToolName="SAM2 masks"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
