FROM nimbusimage/worker-base:latest

# Copy the entrypoint script
COPY ./workers/properties/points/point_to_nearest_blob_distance/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="point" \
      interfaceName="Distance to nearest blob" \
      interfaceCategory="Distance"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]