"""Pre-download the built-in Cellpose-SAM checkpoints at build time.

Instantiating ``CellposeModel(pretrained_model=<name>)`` fetches the weights to
``models.MODELS_DIR`` on first use, so doing it here bakes them into the image
and avoids a multi-GB download on the first run. ``gpu=True`` matches runtime;
on a GPU-less build host cellpose falls back to CPU but still downloads.
"""

from cellpose import models

from models_config import BASE_MODEL_CHECKPOINTS

for checkpoint in set(BASE_MODEL_CHECKPOINTS.values()):
    print(f"Downloading Cellpose-SAM checkpoint: {checkpoint}")
    models.CellposeModel(gpu=True, pretrained_model=checkpoint)
