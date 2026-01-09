FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base
LABEL isUPennContrastWorker=True
LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qy tzdata && \
  apt-get install -qy software-properties-common python3-software-properties && \
  apt-get update && apt-get install -qy \
  build-essential \
  wget \
  python3 \
  libffi-dev \
  libssl-dev \
  libjpeg-dev \
  zlib1g-dev \
  git \
  libpython3-dev \
  # Deconwolf build dependencies
  cmake \
  libfftw3-dev \
  libgsl-dev \
  libpng-dev \
  libtiff-dev \
  libomp-dev \
  # OpenCL for GPU support (generic loader and headers)
  ocl-icd-opencl-dev \
  opencl-headers \
  clinfo && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Register NVIDIA OpenCL driver (ICD file tells loader where to find it)
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

ENV PATH="/root/miniforge3/bin:$PATH"
ARG PATH="/root/miniforge3/bin:$PATH"

RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh

# Clone and build deconwolf with OpenCL support
RUN git clone https://github.com/elgw/deconwolf.git /deconwolf && \
    cd /deconwolf && \
    mkdir builddir && cd builddir && \
    cmake .. && \
    cmake --build . -j$(nproc) && \
    cmake --install . --prefix /usr

FROM base AS build

# Copy and create conda environment
COPY ./workers/annotations/deconwolf/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install NimbusImage annotation client
RUN git clone https://github.com/arjunrajlaboratory/NimbusImage/
RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/

# Install annotation_utilities
COPY ./annotation_utilities /annotation_utilities
RUN pip install /annotation_utilities

# Install large_image for image processing (zarr source supports multi-dimensional images)
RUN pip install large-image-source-zarr large-image-source-tiff large-image-converter large-image --find-links https://girder.github.io/large_image_wheels

# Copy entrypoint
COPY ./workers/annotations/deconwolf/entrypoint.py /

WORKDIR /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Deconwolf Deconvolution" \
      interfaceCategory="Image Processing" \
      description="Deconvolves images using Richardson-Lucy algorithm with Born-Wolf PSF model (GPU-accelerated)" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Deconwolf Deconvolution"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
