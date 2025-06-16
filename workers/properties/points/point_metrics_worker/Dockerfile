FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/points/point_metrics_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="point" \
      interfaceName="Point metrics" \
      interfaceCategory="Morphology" \
      description="Computes a variety of metrics for points, like X and Y coordinates"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python3", "/entrypoint.py"]