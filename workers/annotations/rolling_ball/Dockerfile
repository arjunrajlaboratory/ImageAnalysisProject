FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/rolling_ball/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Rolling Ball" \
      interfaceCategory="Image Processing" \
      description="Corrects images using a rolling ball" \
      hasPreview="True" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Rolling Ball"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]