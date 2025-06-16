FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/blobs/blob_overlap_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="polygon" \
      interfaceName="Blob overlap" \
      interfaceCategory="Relational" \
      description="Compute overlap between blobs"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python3", "/entrypoint.py"]