#!/usr/bin/env bash
# Fast in-process activation of the "worker" conda env, used as the Docker
# ENTRYPOINT in place of `conda run --no-capture-output -n worker`.
#
# Why: `conda run` spawns a *second* Python process (the conda CLI itself) on
# every invocation just to activate the env and exec the target interpreter.
# Because NimbusImage launches a fresh container per job, that wrapper adds
# ~0.8-1.0s of pure startup latency to every single run. This script activates
# the env directly instead: it puts the env on PATH and sources the env's
# activate.d hooks -- which export GDAL_DATA / PROJ_DATA / GDAL_DRIVER_PATH that
# rasterio / large_image rely on at runtime -- then execs the env's python.
# This preserves conda's activation side effects while skipping the extra
# interpreter. See todo/worker-startup-latency.md for measurements and rationale.
#
# Usage in a worker Dockerfile (build context is the repo root):
#   COPY ./workers/base_docker_images/run_worker.sh /usr/local/bin/run_worker.sh
#   RUN chmod +x /usr/local/bin/run_worker.sh
#   ENTRYPOINT ["/usr/local/bin/run_worker.sh", "/entrypoint.py"]
# Args appended by `docker run` (e.g. --apiUrl ... --request ...) are forwarded
# to the entrypoint script unchanged.

# Locate the worker env: miniforge on amd64 (production), miniconda on arm64 (Mac dev).
CONDA_PREFIX="$(ls -d /root/miniforge3/envs/worker /root/miniconda3/envs/worker 2>/dev/null | head -n1)"
export CONDA_PREFIX
export CONDA_DEFAULT_ENV=worker
export PATH="$CONDA_PREFIX/bin:$PATH"

# Source the env's activation hooks (GDAL/PROJ/etc.), exactly as conda would on activate.
if [ -d "$CONDA_PREFIX/etc/conda/activate.d" ]; then
  for _hook in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do
    [ -e "$_hook" ] && . "$_hook"
  done
fi

exec "$CONDA_PREFIX/bin/python" "$@"
