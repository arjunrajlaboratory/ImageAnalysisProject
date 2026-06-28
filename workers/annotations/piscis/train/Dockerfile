FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as base
LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

LABEL isUPennContrastWorker=True
LABEL isAnnotationWorker=True
LABEL interfaceName="Piscis (Train)"
LABEL interfaceCategory="Piscis"

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

ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

FROM base as build

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY ./workers/annotations/piscis/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

RUN git clone https://github.com/arjunrajlaboratory/NimbusImage

RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/


RUN git clone https://github.com/zjniu/Piscis.git
WORKDIR /Piscis
RUN pip install .

WORKDIR /
RUN pip install rasterio shapely

RUN git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/
RUN pip install /ImageAnalysisProject/annotation_utilities
RUN pip install /ImageAnalysisProject/worker_client

COPY ./workers/annotations/piscis/download_models.py /
RUN python /download_models.py

COPY ./workers/annotations/piscis/utils.py /
COPY ./workers/annotations/piscis/train/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isAnnotationWorker=True \
      interfaceName="Piscis spot training" \
      interfaceCategory="Piscis" \
      description="Retrain Piscis to detect your particular spots" \
      defaultToolName="Piscis training"

# Fast env activation in place of `conda run` (see todo/worker-startup-latency.md)
COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]