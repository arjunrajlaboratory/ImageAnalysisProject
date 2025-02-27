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
  libpython3-dev && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# The below is for the x86 and should be changed for other architectures
ENV PATH="/root/miniforge3/bin:$PATH"
ARG PATH="/root/miniforge3/bin:$PATH"

RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh
# END x86 specific

FROM base as build

COPY ./workers/properties/blobs/blob_annulus_intensity_worker/environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Install the annotation utilities
COPY ./annotation_utilities /annotation_utilities
WORKDIR /annotation_utilities
RUN pip install .

WORKDIR /
RUN git clone https://github.com/Kitware/UPennContrast/
WORKDIR /UPennContrast

RUN pip install -r devops/girder/annotation_client/requirements.txt
RUN pip install -e devops/girder/annotation_client/

COPY ./workers/properties/blobs/blob_annulus_intensity_worker/entrypoint.py /

LABEL isUPennContrastWorker="" \
      isPropertyWorker="" \
      annotationShape="polygon" \
      interfaceName="Annulus intensity measurements" \
      interfaceCategory="Intensity" \
      description="Compute intensity measurements (mean, median, etc.) for an annular region around blobs"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python3", "/entrypoint.py"]