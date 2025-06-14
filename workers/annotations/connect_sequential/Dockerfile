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

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]