FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/points/point_circle_intensity_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="point" \
      interfaceName="Point intensity" \
      interfaceCategory="Intensity" \
      description="Computes a variety of intensity metrics around a point; radius 1 for pixel precision"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)