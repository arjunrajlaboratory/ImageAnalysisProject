FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/blobs/blob_overlap_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="polygon" \
      interfaceName="Blob overlap" \
      interfaceCategory="Relational" \
      description="Compute overlap between blobs"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)