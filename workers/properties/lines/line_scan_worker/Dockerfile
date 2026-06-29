FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/lines/line_scan_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="line" \
      interfaceName="Line scan CSV" \
      interfaceCategory="Intensity" \
      description="Compute the intensity along a line and puts the results in a CSV file"

# ENTRYPOINT inherited from worker-base (run_worker.sh /entrypoint.py — see todo/worker-startup-latency.md)
