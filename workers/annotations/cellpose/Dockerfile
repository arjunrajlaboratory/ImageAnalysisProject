FROM ubuntu:bionic as base

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
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-aarch64.sh -b \
    && rm -f Miniforge3-Linux-aarch64.sh

FROM base as build

COPY ./environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

RUN git clone https://github.com/Kitware/UPennContrast/
WORKDIR /UPennContrast
RUN git pull origin add-annotation-worker

RUN pip install -r devops/girder/annotation_client/requirements.txt
RUN pip install -e devops/girder/annotation_client/
RUN pip install cellpose --no-deps

COPY ./download_models.py /
RUN python /download_models.py

COPY ./entrypoint.py /

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]