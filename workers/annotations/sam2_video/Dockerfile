FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 as base
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
  r-base \
  libffi-dev \
  libssl-dev \
  libjpeg-dev \
  zlib1g-dev \
  r-base \
  git \
  libpython3-dev && \
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

RUN pip install rtree shapely

RUN git clone https://github.com/Kitware/UPennContrast/

RUN pip install -r /UPennContrast/devops/girder/annotation_client/requirements.txt
RUN pip install -e /UPennContrast/devops/girder/annotation_client/

RUN git clone -b nimbus-video-predictor https://github.com/arjunrajlaboratory/segment-anything-2-nimbus.git
RUN pip install -e /segment-anything-2-nimbus
RUN /segment-anything-2-nimbus/checkpoints/download_ckpts.sh

COPY ./utils.py /
COPY ./entrypoint.py /

RUN git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/
RUN pip install /ImageAnalysisProject/annotation_utilities

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="SAM2 propagator" \
      interfaceCategory="SAM2" \
      description="Uses SAM2 to track video through time or Z" \
      annotationShape="polygon"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]