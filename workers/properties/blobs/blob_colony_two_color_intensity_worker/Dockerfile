FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/blobs/blob_colony_two_color_intensity_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="polygon" \
      interfaceName="Colony two color intensity" \
      interfaceCategory="Intensity" \
      description="Compute intensity across two channels for blobs; useful for colony intensity measurements"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)
