FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/blobs/blob_intensity_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="polygon" \
      interfaceName="Blob intensity measurements" \
      interfaceCategory="Intensity" \
      description="Compute intensity measurements (mean, median, etc.) for blobs"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python3", "/entrypoint.py"]