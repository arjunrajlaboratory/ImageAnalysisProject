"""Built-in Cellpose-SAM model options for the cellposesam worker.

Maps the human-friendly names shown in the Model dropdown to the cellpose
checkpoint identifiers passed to ``CellposeModel(pretrained_model=...)``.

Kept deliberately import-free (no cellpose/deeptile/annotation_client) so the
mapping can be unit-tested in the lightweight local venv without the full
worker stack.
"""

from pathlib import Path


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


# The diameter at which cellpose applies no rescaling. ``CellposeModel.eval()``
# computes ``rescale = 30. / diameter`` with 30 as a hardcoded literal
# (models.py:269) -- it is not a per-model ``diam_mean``, which v4 ignores
# entirely. So 30 yields rescale == 1.0, exactly what ``diameter=None`` yields,
# and it serves as the interface default: an in-range, self-describing "leave it
# alone" value rather than an out-of-band 0 sentinel.
DEFAULT_DIAMETER = 30.0

# Smallest diameter the interface offers. A small diameter *enlarges* the image
# (30/10 = 3x), so this bounds how far a tile can blow up before inference.
MIN_DIAMETER = 10.0

# Largest diameter the interface offers; shrinks the image by 30/200 = 0.15x.
MAX_DIAMETER = 200.0


def diameter_rescale(diameter):
    """Return the factor cellpose will resize the image by for ``diameter``.

    Mirrors ``CellposeModel.eval()`` in cellpose 4.2.1.1, which sets
    ``rescale = 30. / diameter`` only when ``diameter is not None and
    diameter > 0`` and otherwise leaves it at 1.0. Exposed so the worker can
    flag a rescale that would blow the tile size up past what the GPU can hold,
    before loading the model.
    """
    if diameter is None or float(diameter) <= 0:
        return 1.0
    return 30.0 / float(diameter)


def build_cellpose_parameters(model_name, models_dir, diameter=None):
    """Build the Cellpose-SAM constructor and evaluation parameters.

    ``diameter`` is forwarded to ``CellposeModel.eval()``, which rescales the
    image by ``30 / diameter`` so objects land near the 30 px scale the network
    expects. It is omitted entirely whenever it would be a no-op -- at the
    identity value ``DEFAULT_DIAMETER``, and for the None/0/negative values that
    cellpose itself ignores -- so the default run issues exactly the same call
    the worker made while the field was absent.

    Note that only the *constructor* argument ``diam_mean`` was deprecated in
    cellpose v4.0.1+; the eval-time ``diameter`` is still honoured.
    """
    if model_name in BASE_MODEL_CHECKPOINTS:
        pretrained_model = BASE_MODEL_CHECKPOINTS[model_name]
    else:
        pretrained_model = str(Path(models_dir) / model_name)

    eval_parameters = {}
    # Keyed off the resulting rescale rather than the raw value, so anything
    # cellpose would treat as identity is dropped here instead of being passed
    # through to no effect.
    if diameter_rescale(diameter) != 1.0:
        eval_parameters['diameter'] = float(diameter)

    return {
        'model_parameters': {
            'gpu': True,
            'pretrained_model': pretrained_model,
        },
        'eval_parameters': eval_parameters,
    }
