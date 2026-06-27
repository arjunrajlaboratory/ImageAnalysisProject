FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/blobs/blob_metrics_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="polygon" \
      interfaceName="Blob metrics" \
      interfaceCategory="Morphology" \
      description="Compute area, perimeter, and other metrics for blobs"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)