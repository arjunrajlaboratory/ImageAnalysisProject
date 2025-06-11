FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/gaussian_blur/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Gaussian Blur" \
      interfaceCategory="Image Processing" \
      description="Applies Gaussian blur to images" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Gaussian Blur"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]