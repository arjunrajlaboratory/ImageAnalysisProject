"""Unit tests for the built-in Cellpose-SAM model mapping.

These exercise ``models_config`` in isolation — it must stay free of heavy
imports (cellpose/deeptile/annotation_client) so it runs in the lightweight
local venv without the full worker stack. Run with:

    .cache/testvenv/bin/pytest workers/annotations/cellposesam/tests -q
"""

import sys
from pathlib import Path

# Put the worker directory (parent of tests/) on the path so we can import the
# standalone mapping module without installing the whole worker.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models_config  # noqa: E402


def test_default_model_resolves_to_cpsam_v2():
    """The default dropdown selection must run the new cpsam_v2 checkpoint."""
    checkpoint = models_config.BASE_MODEL_CHECKPOINTS[models_config.DEFAULT_MODEL]
    assert checkpoint == 'cpsam_v2'


def test_legacy_label_resolves_to_original_cpsam():
    """The legacy option must still map to the original April 2025 cpsam checkpoint."""
    assert models_config.BASE_MODEL_CHECKPOINTS['cellpose-sam (legacy cpsam)'] == 'cpsam'


def test_default_model_is_a_selectable_base_model():
    """The configured default must be one of the offered base models."""
    assert models_config.DEFAULT_MODEL in models_config.BASE_MODELS


def test_base_models_offers_both_builtins():
    """Both the v2 default and the legacy option must remain available."""
    assert set(models_config.BASE_MODELS) == {
        'cellpose-sam',
        'cellpose-sam (legacy cpsam)',
    }
