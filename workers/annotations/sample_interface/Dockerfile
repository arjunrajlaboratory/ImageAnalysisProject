FROM nimbusimage/test-worker-base:latest

# Copy the entrypoint script
COPY --chown=$MAMBA_USER:$MAMBA_USER ./workers/annotations/sample_interface/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Sample interface" \
      interfaceCategory="Testing" \
      description="Demonstrates all interface types, messaging patterns, and batch mode"

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/entrypoint.py"]
