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

# Keep in mind that the below only works on the architecture that the docker image is built on, and not the architecture that it is run on.
# Conditionally download and install conda/miniforge based on architecture
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        echo "Running ARM 64 specific commands" && \
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh && \
        mkdir -p /root/.conda && \
        bash Miniconda3-latest-Linux-aarch64.sh -b && \
        rm -f Miniconda3-latest-Linux-aarch64.sh; \
    elif [ "$(uname -m)" = "x86_64" ]; then \
        echo "Running AMD 64 specific commands" && \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
        mkdir -p /root/.conda && \
        bash Miniforge3-Linux-x86_64.sh -b && \
        rm -f Miniforge3-Linux-x86_64.sh; \
    else \
        echo "Unsupported architecture"; \
        exit 1; \
    fi


# Set the PATH environment variable
ENV PATH="/root/miniconda3/bin:/root/miniforge3/bin:$PATH"

# Create conda environment from environment.yml
COPY ./environment.yml /
RUN conda env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]





RUN git clone https://github.com/Kitware/UPennContrast/
WORKDIR /UPennContrast

RUN pip install -r devops/girder/annotation_client/requirements.txt
RUN pip install -e devops/girder/annotation_client/

COPY ./entrypoint.py /

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python3", "/entrypoint.py"]