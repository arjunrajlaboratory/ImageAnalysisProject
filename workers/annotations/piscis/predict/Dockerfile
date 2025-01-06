FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as base
LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

LABEL isUPennContrastWorker=True
LABEL isAnnotationWorker=True
LABEL interfaceName="Piscis (Predict)"
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
    https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py39_4.12.0-Linux-x86_64.sh

FROM base as build

COPY ./environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

RUN git clone https://github.com/Kitware/UPennContrast/

RUN pip install -r /UPennContrast/devops/girder/annotation_client/requirements.txt
RUN pip install -e /UPennContrast/devops/girder/annotation_client/
RUN pip install -U "jax[cuda12]" piscis
# RUN pip install -U "jax[cuda12_pip]==0.4.26" "jaxlib==0.4.26+cuda12.cudnn89" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install piscis

RUN git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/
RUN pip install /ImageAnalysisProject/annotation_utilities
RUN pip install /ImageAnalysisProject/worker_client

COPY ./download_models.py /
RUN python /download_models.py

COPY ./utils.py /
COPY ./predict/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isAnnotationWorker=True \
      interfaceName="Piscis spot detection" \
      interfaceCategory="Piscis" \
      description="Uses Piscis to find spots" \
      defaultToolName="Spots"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]