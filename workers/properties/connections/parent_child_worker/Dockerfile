FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/connections/parent_child_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="any" \
      interfaceName="Connection IDs" \
      interfaceCategory="Connections" \
      description="Creates a set of identifiers that indicate the parent-child relationships between polygons"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python3", "/entrypoint.py"]