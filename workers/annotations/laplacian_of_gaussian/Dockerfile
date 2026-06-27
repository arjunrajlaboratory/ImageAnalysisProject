FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/annotations/laplacian_of_gaussian/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Laplacian of Gaussian" \
      interfaceCategory="Object Detection" \
      description="Detects spots in images using the Laplacian of Gaussian method" \
      hasPreview="True"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)
