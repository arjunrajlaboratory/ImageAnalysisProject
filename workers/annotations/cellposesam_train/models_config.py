"""Built-in Cellpose-SAM model options for the cellposesam_train worker.

Maps the human-friendly names shown in the Base Model dropdown to the cellpose
checkpoint identifiers passed to ``CellposeModel(pretrained_model=...)`` as the
starting point for fine-tuning.

Kept in sync with the cellposesam (inference) worker's ``models_config.py`` so
the base-model choices match. Kept deliberately import-free (no
cellpose/annotation_client) so the mapping can be unit-tested in the lightweight
local venv without the full worker stack.
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
    """Build the sorted Base Model-dropdown options from the built-in labels plus
    custom Girder model names.

    The built-in labels are reserved: a custom Girder model whose name collides
    with one is dropped, because ``compute()`` routes any name in ``BASE_MODELS``
    to the built-in checkpoint — so a same-named custom model could never be
    loaded and would only create a confusing duplicate entry.
    """
    custom = [name for name in girder_model_names
              if name not in BASE_MODEL_CHECKPOINTS]
    return sorted(set(BASE_MODELS) | set(custom))


def validate_output_model_name(model_name):
    """Return a normalized custom model name or raise ``ValueError``.

    Built-in dropdown labels are reserved because inference always routes them
    to their bundled checkpoints, making a custom model with the same name
    impossible to select.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Please provide a name for the retrained model.")

    model_name = model_name.strip()
    if model_name in BASE_MODEL_CHECKPOINTS:
        raise ValueError(
            f'"{model_name}" is reserved for a built-in Cellpose-SAM model. '
            "Please choose a different output model name."
        )
    return model_name
