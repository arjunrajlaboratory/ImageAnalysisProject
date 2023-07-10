
FROM ghcr.io/arjunrajlaboratory/base_m1_image:latest

# Here, the idea is to derive from the base image, and then install the dependencies. We use update instead of create.
COPY ./environment.yml /
RUN conda env update --file /environment.yml
SHELL ["conda", "run", "-n", "worker", "/bin/bash", "-c"]

COPY ./entrypoint.py /

LABEL isUPennContrastWorker="" \
      isAnnotationWorker="" \
      interfaceName="Random square" \
      interfaceCategory="random"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "worker", "python", "/entrypoint.py"]