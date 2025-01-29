FROM ubuntu:jammy as base
LABEL isUPennContrastWorker=True

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -qy tzdata && \
  apt-get install -qy software-properties-common python3-software-properties && \
  apt-get update && apt-get install -qy \
  build-essential \
  wget \
  python3 \
  r-base \
  libffi-dev \
  libssl-dev \
  libjpeg-dev \
  zlib1g-dev \
  r-base \
  git \
  libpython3-dev \
  libtiff-dev \
  libvips-dev \
  libopenjp2-7-dev \
  libwebp-dev \
  openslide-tools \
  python3-openslide \
  && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniforge3/bin:$PATH"
ARG PATH="/root/miniforge3/bin:$PATH"

RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh

FROM base as build

COPY ./environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

RUN git clone https://github.com/arjunrajlaboratory/NimbusImage/
RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/

RUN git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/
RUN pip install /ImageAnalysisProject/annotation_utilities
RUN pip install /ImageAnalysisProject/worker_client

COPY ./entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Gaussian Blur" \
      interfaceCategory="Image Processing" \
      description="Blurs images using a Gaussian kernel" \
      hasPreview="True" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Gaussian Blur"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]