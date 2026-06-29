FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/histogram_matching/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Histogram Matching" \
      interfaceCategory="Image Processing" \
      description="Corrects images using histogram matching" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Histogram Matching"

# ENTRYPOINT inherited from image-processing-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)