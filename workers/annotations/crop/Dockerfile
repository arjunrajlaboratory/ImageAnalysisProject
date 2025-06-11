FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/crop/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Crop" \
      interfaceCategory="Image Processing" \
      description="Crops images, allowing you to drop particular slices of data and crop to blobs or rectangles" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Crop"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]