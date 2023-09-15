FROM ubuntu:jammy as base
LABEL isUPennContrastWorker=True
ARG CHIP_TYPE=default # Default to x86
ARG INSTALLER_PATH="/root/miniforge3/bin"
ARG INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

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

# Set values for AppleSilicon
RUN if [ "$CHIP_TYPE" = "AppleSilicon" ]; then \
      export INSTALLER_PATH="/root/miniconda3/bin"; \
      export INSTALLER_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    fi

# Set the PATH ENV
ENV PATH="${INSTALLER_PATH}:$PATH"
ARG PATH="${INSTALLER_PATH}:$PATH"

# Install the required package
RUN wget $INSTALLER_URL \
    && mkdir /root/.conda \
    && bash $(basename $INSTALLER_URL) -b \
    && rm -f $(basename $INSTALLER_URL)

FROM base as build

COPY ./environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install the annotation utilities
RUN git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/
WORKDIR /ImageAnalysisProject
RUN pip install ./annotation_utilities

WORKDIR /
RUN git clone https://github.com/Kitware/UPennContrast/
WORKDIR /UPennContrast

RUN pip install -r devops/girder/annotation_client/requirements.txt
RUN pip install -e devops/girder/annotation_client/


#COPY ./annotation_tools.py /
#COPY ./batch_argument_parser.py /
COPY ./entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Connect to Nearest" \
      interfaceCategory="Connections" \
      description="Connects objects to each other based on distance"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]

