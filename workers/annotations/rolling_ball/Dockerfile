FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/rolling_ball/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Rolling Ball" \
      interfaceCategory="Image Processing" \
      description="Corrects images using a rolling ball" \
      hasPreview="True" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Rolling Ball"

# ENTRYPOINT inherited from image-processing-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)