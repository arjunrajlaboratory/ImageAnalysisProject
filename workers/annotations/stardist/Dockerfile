FROM nimbusimage/cuda-ml-worker-base:latest

# Stardist requires specific pinned package versions, so recreate the environment
COPY ./workers/annotations/stardist/environment.yml /
RUN mamba env remove -n worker -y && mamba env create --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

# Reinstall clients into the new environment
RUN pip install -r /NimbusImage/devops/girder/annotation_client/requirements.txt
RUN pip install -e /NimbusImage/devops/girder/annotation_client/
RUN pip install /annotation_utilities
RUN pip install /worker_client

# Install DeepTile for tiling support
RUN git clone https://github.com/arjunrajlaboratory/DeepTile
RUN pip install -e /DeepTile
RUN pip install rtree shapely

# Download models
COPY ./workers/annotations/stardist/download_models.py /
RUN python /download_models.py

COPY ./workers/annotations/stardist/utils.py /
COPY ./workers/annotations/stardist/entrypoint.py /

LABEL isUPennContrastWorker=True \
      isAnnotationWorker=True \
      interfaceName="Stardist" \
      interfaceCategory="Stardist" \
      annotationShape="polygon" \
      description="Uses Stardist to find cells and nuclei" \
      defaultToolName="Stardist"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]
