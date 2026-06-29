FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/connections/children_count_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="polygon" \
      interfaceName="Count connected objects" \
      interfaceCategory="Quantification" \
      description="Count the number of children objects that are connected to a parent polygon"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)