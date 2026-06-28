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

RUN git clone https://github.com/Kitware/UPennContrast/

RUN pip install -r /UPennContrast/devops/girder/annotation_client/requirements.txt
RUN pip install -e /UPennContrast/devops/girder/annotation_client/

RUN pip install piscis==1.0.0
RUN pip install flax
# Install a CUDA-enabled jax so Piscis runs on the GPU. `pip install flax` above
# pulls the CPU-only jaxlib by default; reinstall jax with the cuda12 extra pinned
# to the version flax resolved, adding the GPU backend without drifting the version.
RUN pip install --upgrade "jax[cuda12]==$(python -c 'import jax; print(jax.__version__)')"

WORKDIR /
RUN git clone https://github.com/arjunrajlaboratory/ImageAnalysisProject/
RUN pip install /ImageAnalysisProject/annotation_utilities
RUN pip install /ImageAnalysisProject/worker_client

COPY ./workers/annotations/piscis/download_models.py /
RUN python /download_models.py

COPY ./workers/annotations/piscis/utils.py /
COPY ./workers/annotations/piscis/predict/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isAnnotationWorker=True \
      interfaceName="Piscis spot detection" \
      interfaceCategory="Piscis" \
      description="Uses Piscis to find spots" \
      defaultToolName="Spots"

# Fast env activation in place of `conda run` (see todo/worker-startup-latency.md)
COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
RUN chmod +x /usr/local/bin/run_worker.sh
ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]