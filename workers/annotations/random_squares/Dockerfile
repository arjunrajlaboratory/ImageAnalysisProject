FROM nimbusimage/test-worker-base:latest

# Copy the entrypoint script
COPY --chown=$MAMBA_USER:$MAMBA_USER ./workers/annotations/random_squares/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Random squares" \
      interfaceCategory="Testing" \
      description="Generates random square polygon annotations for testing"

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/entrypoint.py"]
