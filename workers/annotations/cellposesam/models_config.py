"""Built-in Cellpose-SAM model options for the cellposesam worker.

Maps the human-friendly names shown in the Model dropdown to the cellpose
checkpoint identifiers passed to ``CellposeModel(pretrained_model=...)``.

Kept deliberately import-free (no cellpose/deeptile/annotation_client) so the
mapping can be unit-tested in the lightweight local venv without the full
worker stack.
"""

# Human-friendly dropdown label -> cellpose built-in checkpoint name.
# Insertion order lists the default (cpsam_v2) first.
BASE_MODEL_CHECKPOINTS = {
    # New default: SAM-ViTL backbone, released June 2026; fixes spurious masks
    # in low-contrast regions relative to the original cpsam.
    'cellpose-sam': 'cpsam_v2',
    # Original Cellpose-SAM model, released April 2025. Kept selectable so prior
    # results remain reproducible.
    'cellpose-sam (legacy cpsam)': 'cpsam',
}

# The dropdown option selected by default.
DEFAULT_MODEL = 'cellpose-sam'

# The list of built-in model labels offered in the interface.
BASE_MODELS = list(BASE_MODEL_CHECKPOINTS)


def build_model_items(girder_model_names):
    """Build the sorted Model-dropdown options from the built-in labels plus
    custom Girder model names.

    The built-in labels are reserved: a custom Girder model whose name collides
    with one is dropped, because ``compute()`` routes any name in ``BASE_MODELS``
    to the built-in checkpoint — so a same-named custom model could never be
    loaded and would only create a confusing duplicate entry.
    """
    custom = [name for name in girder_model_names
              if name not in BASE_MODEL_CHECKPOINTS]
    return sorted(set(BASE_MODELS) | set(custom))
