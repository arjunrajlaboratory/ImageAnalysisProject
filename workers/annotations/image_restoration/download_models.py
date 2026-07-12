"""Pre-download restoration model weights at build time.

- Cellpose3 restoration checkpoints (denoise/deblur/upsample) are downloaded
  unconditionally, mirroring cellposesam's download_models.py: instantiating
  ``denoise.DenoiseModel(model_type=<name>)`` fetches weights to Cellpose's
  cache dir on first use, so doing it here bakes them into the image and
  avoids a multi-GB download on the first job run.

- FluoResFM weights are downloaded **best-effort only**. The FluoResFM
  authors distribute the pretrained checkpoint via Google Drive / Baidu Yun
  (see https://github.com/qiqi-lu/fluoresfm), not a stable, scriptable direct
  -download URL, so there is no reliable way to fetch it unattended at build
  time. If ``FLUORESFM_WEIGHTS_URL`` is provided as a build-time environment
  variable pointing to a mirror you control (e.g. an internal artifact store
  or a Zenodo/HF mirror you've set up), this script will try to fetch it into
  ``/opt/fluoresfm_weights/fluoresfm.pt``. Any failure here is caught and
  logged -- it must NOT fail the Docker build. At runtime, entrypoint.py reads
  the checkpoint path from the ``FLUORESFM_WEIGHTS`` environment variable
  (see ``resolve_fluoresfm_weights()``), which the Dockerfile can default to
  this path, and gracefully ``sendError``s with instructions if it is
  missing/unset.
"""

import os
import urllib.request

from models_config import CELLPOSE3_RESTORATION_CHECKPOINTS

print("Downloading Cellpose3 restoration checkpoints...")
try:
    from cellpose import denoise

    for checkpoint in CELLPOSE3_RESTORATION_CHECKPOINTS:
        print(f"  Downloading Cellpose3 restoration checkpoint: {checkpoint}")
        denoise.DenoiseModel(model_type=checkpoint, gpu=False)
except Exception as e:  # pragma: no cover - build-time diagnostic only
    print(f"  WARNING: failed to pre-download one or more Cellpose3 checkpoints: {e}")
    print("  These will instead be downloaded on first use at runtime.")

print("Attempting best-effort FluoResFM weight download...")
fluoresfm_weights_url = os.environ.get('FLUORESFM_WEIGHTS_URL', '').strip()
if not fluoresfm_weights_url:
    print(
        "  FLUORESFM_WEIGHTS_URL not set at build time -- skipping. FluoResFM "
        "weights are distributed via Google Drive/Baidu Yun "
        "(https://github.com/qiqi-lu/fluoresfm) and must be supplied at "
        "runtime via the FLUORESFM_WEIGHTS environment variable/volume mount. "
        "This is expected and does not fail the build."
    )
else:
    try:
        os.makedirs('/opt/fluoresfm_weights', exist_ok=True)
        dest = '/opt/fluoresfm_weights/fluoresfm.pt'
        print(f"  Downloading FluoResFM weights from {fluoresfm_weights_url} -> {dest}")
        urllib.request.urlretrieve(fluoresfm_weights_url, dest)
        print("  FluoResFM weights downloaded successfully.")
    except Exception as e:  # pragma: no cover - best-effort, must not fail the build
        print(f"  WARNING: FluoResFM weight download failed ({e}); continuing without it.")
        print("  Set FLUORESFM_WEIGHTS at runtime to a mounted checkpoint instead.")

print("download_models.py complete.")
