FROM nimbusimage/test-worker-base:latest

# Copy the entrypoint script
COPY --chown=$MAMBA_USER:$MAMBA_USER ./workers/annotations/error_generator/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Error generator" \
      interfaceCategory="Testing" \
      description="Generates various error conditions for testing frontend error reporting"

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/entrypoint.py"]
