FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as base
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

RUN git clone -b fix-cellpose https://github.com/arjunrajlaboratory/DeepTile
RUN pip install -e /DeepTile
RUN pip install rtree shapely

RUN git clone https://github.com/Kitware/UPennContrast/

RUN pip install -r /UPennContrast/devops/girder/annotation_client/requirements.txt
RUN pip install -e /UPennContrast/devops/girder/annotation_client/

COPY ./download_models.py /
RUN python /download_models.py

COPY ./utils.py /
COPY ./girder_utils.py /
COPY ./entrypoint.py /

RUN git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/
RUN pip install /ImageAnalysisProject/annotation_utilities
RUN pip install /ImageAnalysisProject/worker_client

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Cellpose retrain" \
      interfaceCategory="Cellpose" \
      description="Retrains Cellpose models based on user-selected images" \
      advancedOptionsPanel="False" \
      annotationConfigurationPanel="False" \
      defaultToolName="Cellpose retrain" \
      annotationShape="polygon"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]