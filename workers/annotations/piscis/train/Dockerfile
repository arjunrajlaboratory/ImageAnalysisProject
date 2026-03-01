FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base
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

FROM base AS build

COPY ./environment.yml /
RUN mamba env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

RUN git clone https://github.com/arjunrajlaboratory/NimbusImage/
RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/

RUN git clone https://github.com/zjniu/Piscis.git
WORKDIR /Piscis
RUN pip install .
RUN pip install flax

WORKDIR /
RUN pip install rasterio shapely

RUN git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/
RUN pip install /ImageAnalysisProject/annotation_utilities
RUN pip install /ImageAnalysisProject/worker_client

COPY ./download_models.py /
RUN python /download_models.py

COPY ./utils.py /
COPY ./train/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isAnnotationWorker=True \
      interfaceName="Piscis spot training" \
      interfaceCategory="Piscis" \
      description="Retrain Piscis to detect your particular spots" \
      defaultToolName="Piscis training"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
