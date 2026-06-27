FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/registration/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Time lapse registration" \
      interfaceCategory="Image Processing" \
      description="Corrects images using time lapse registration" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Time lapse registration"

# ENTRYPOINT inherited from image-processing-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)