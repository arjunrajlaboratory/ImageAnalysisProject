FROM nimbusimage/image-processing-base:latest

# Install deconwolf build dependencies (including OpenCL for GPU support)
RUN apt-get update && apt-get install -y \
    cmake \
    libfftw3-dev \
    libgsl-dev \
    libpng-dev \
    libomp-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    && rm -rf /var/lib/apt/lists/*

# Clone and build deconwolf
RUN git clone https://github.com/elgw/deconwolf.git /deconwolf && \
    cd /deconwolf && \
    mkdir builddir && cd builddir && \
    cmake .. && \
    cmake --build . -j$(nproc) && \
    cmake --install . --prefix /usr

COPY ./workers/annotations/deconwolf/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Deconwolf Deconvolution" \
      interfaceCategory="Image Processing" \
      description="Deconvolves images using Richardson-Lucy algorithm with Born-Wolf PSF model" \
      hasPreview="False" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Deconwolf Deconvolution"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
