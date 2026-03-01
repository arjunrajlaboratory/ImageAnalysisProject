FROM nimbusimage/cuda-ml-worker-base:latest

SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install PyTorch with CUDA 11.8 support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM1 and dependencies
RUN pip install segment-anything pycocotools onnxruntime onnx

# Download SAM ViT-H checkpoint
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY ./workers/annotations/sam_fewshot_segmentation/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="SAM few-shot segmentation" \
      interfaceCategory="SAM" \
      description="Uses SAM1 ViT-H features for few-shot segmentation based on training annotations" \
      annotationShape="polygon" \
      defaultToolName="SAM few-shot"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
