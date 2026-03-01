FROM nimbusimage/cuda-ml-worker-base:latest

SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install PyTorch with CUDA 11.8 support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM1 and dependencies
RUN pip install segment-anything pycocotools onnxruntime onnx

# Download SAM ViT-H checkpoint
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY ./workers/annotations/sam_automatic_mask_generator/utils.py /
COPY ./workers/annotations/sam_automatic_mask_generator/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="SAM automatic mask generator" \
      interfaceCategory="SAM" \
      description="Uses SAM to find all masks in the image" \
      annotationShape="polygon"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
