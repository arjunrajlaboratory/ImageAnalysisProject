FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/annotations/connect_to_nearest/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Connect to Nearest" \
      interfaceCategory="Connections" \
      description="Connects objects to each other based on distance" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Connect to Nearest"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]

