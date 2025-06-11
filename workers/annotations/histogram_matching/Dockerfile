FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/histogram_matching/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Histogram Matching" \
      interfaceCategory="Image Processing" \
      description="Corrects images using histogram matching" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Histogram Matching"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]