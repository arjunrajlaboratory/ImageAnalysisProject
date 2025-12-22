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

FROM base as build

# Copy and create conda environment
COPY ./workers/annotations/condensatenet_worker/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install NimbusImage annotation client
RUN git clone https://github.com/arjunrajlaboratory/NimbusImage/
RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/

# Install annotation_utilities
COPY ./annotation_utilities /annotation_utilities
RUN pip install /annotation_utilities

# Install CondensateNet package
RUN pip install git+https://github.com/arjunrajlaboratory/condensatenet.git

# Download CondensateNet model (bake into image for faster startup)
RUN python -m condensatenet download --output /models/condensatenet

# Copy entrypoint
COPY ./workers/annotations/condensatenet_worker/entrypoint.py /

WORKDIR /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="CondensateNet segmentation" \
      interfaceCategory="CondensateNet" \
      description="Segments biomolecular condensates using CondensateNet deep learning model" \
      annotationShape="polygon" \
      defaultToolName="Condensates"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
