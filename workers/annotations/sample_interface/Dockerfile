FROM nimbusimage/worker-base:latest

# Install worker_client for batch processing
COPY ./worker_client /worker_client
WORKDIR /worker_client
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]
RUN pip install .
WORKDIR /

# Copy the entrypoint script
COPY ./workers/annotations/sample_interface/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Sample interface" \
      interfaceCategory="Testing" \
      description="Demonstrates all interface types, messaging patterns, and batch mode"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
