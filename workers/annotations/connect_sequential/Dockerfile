FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/annotations/connect_sequential/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Connect Sequential" \
      interfaceCategory="Connections" \
      description="Connects objects sequentially across time or z-slices" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Connect Sequential"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)