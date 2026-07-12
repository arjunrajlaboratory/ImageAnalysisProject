"""Single source of truth for the Cellpose3 restoration checkpoints this
worker exposes via the 'Cellpose3 model' select (see entrypoint.py) and
pre-downloads at build time (see download_models.py).
"""

CELLPOSE3_RESTORATION_CHECKPOINTS = [
    'denoise_cyto3',
    'deblur_cyto3',
    'upsample_cyto3',
    'denoise_nuclei',
    'deblur_nuclei',
    'oneclick_cyto3',
]
