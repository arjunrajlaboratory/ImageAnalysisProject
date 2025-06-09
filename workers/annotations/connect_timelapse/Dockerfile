FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/annotations/connect_timelapse/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Connect Time Lapse" \
      interfaceCategory="Connections" \
      description="Connects objects across time slices" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Connect Time Lapse"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]

