FROM nimbusimage/image-processing-base:latest

COPY ./workers/annotations/h_and_e_deconvolution/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="H&E Deconvolution" \
      interfaceCategory="Image Processing" \
      description="Deconvolves H&E stains" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="H&E Deconvolution"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]