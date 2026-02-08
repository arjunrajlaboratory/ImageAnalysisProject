FROM nimbusimage/worker-base:latest

# Install worker_client for batch processing
COPY ./worker_client /worker_client
WORKDIR /worker_client
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]
RUN pip install .
WORKDIR /

# Copy the entrypoint script
COPY ./workers/annotations/random_squares/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Random squares" \
      interfaceCategory="Testing" \
      description="Generates random square polygon annotations for testing"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
