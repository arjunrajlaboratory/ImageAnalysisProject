"""Unit tests for the built-in Cellpose-SAM model mapping.

These exercise ``models_config`` in isolation — it must stay free of heavy
imports (cellpose/deeptile/annotation_client) so it runs in the lightweight
local venv without the full worker stack. Run with:

    .cache/testvenv/bin/pytest workers/annotations/cellposesam/tests -q
"""

import importlib.util
from pathlib import Path

# Load under a worker-specific module name so this suite can be collected in
# the same pytest process as cellposesam_train's models_config tests.
_MODELS_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'models_config.py'
_SPEC = importlib.util.spec_from_file_location(
    'cellposesam_models_config', _MODELS_CONFIG_PATH)
models_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(models_config)


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


def test_build_model_items_includes_base_and_custom():
    """Custom Girder model names appear alongside the built-in labels."""
    items = models_config.build_model_items(['my custom model'])
    assert 'cellpose-sam' in items
    assert 'cellpose-sam (legacy cpsam)' in items
    assert 'my custom model' in items


def test_build_model_items_excludes_reserved_name_collision():
    """A custom model named exactly like a base label is dropped, not duplicated.

    Otherwise it would silently route to the built-in checkpoint in compute()
    and the custom weights would never be used.
    """
    items = models_config.build_model_items(
        ['cellpose-sam', 'cellpose-sam (legacy cpsam)'])
    assert items.count('cellpose-sam') == 1
    assert items.count('cellpose-sam (legacy cpsam)') == 1
    assert set(items) == set(models_config.BASE_MODELS)


def test_build_model_items_sorted_and_deduped():
    """Output is sorted and free of duplicates."""
    items = models_config.build_model_items(['zeta', 'alpha', 'alpha'])
    assert items == sorted(set(items))
    assert items.count('alpha') == 1


def test_build_model_items_empty_returns_base_models():
    """With no custom models, only the built-in labels are offered."""
    items = models_config.build_model_items([])
    assert set(items) == set(models_config.BASE_MODELS)
    assert models_config.DEFAULT_MODEL in items


def test_builtin_runtime_parameters_use_checkpoint_at_native_scale(tmp_path):
    parameters = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path)

    assert parameters == {
        'model_parameters': {
            'gpu': True,
            'pretrained_model': 'cpsam_v2',
        },
        'eval_parameters': {},
    }


def test_custom_runtime_parameters_use_downloaded_path_at_native_scale(tmp_path):
    parameters = models_config.build_cellpose_parameters(
        'my custom model', tmp_path)

    assert parameters == {
        'model_parameters': {
            'gpu': True,
            'pretrained_model': str(tmp_path / 'my custom model'),
        },
        'eval_parameters': {},
    }


# --- Diameter / eval-time rescaling -----------------------------------------
#
# cellpose 4.2.1.1 still honours ``diameter`` in ``CellposeModel.eval()``
# (models.py: ``if diameter is not None and diameter > 0: rescale = 30./diameter``),
# and deeptile forwards our ``eval_parameters`` straight into that call. Only the
# constructor argument ``diam_mean`` was deprecated in v4.0.1+. The parameter was
# removed from the worker in a3e4524 on the mistaken assumption that Cellpose-SAM
# ignores it entirely; these tests pin the restored behaviour.


def test_diameter_omitted_keeps_native_resolution(tmp_path):
    """No diameter argument must reproduce the pre-restore native-scale behaviour."""
    parameters = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path)

    assert parameters['eval_parameters'] == {}


def test_diameter_none_keeps_native_resolution(tmp_path):
    """An explicit None is the 'off' value, matching cellpose's own CLI default."""
    parameters = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path, diameter=None)

    assert parameters['eval_parameters'] == {}


def test_diameter_zero_keeps_native_resolution(tmp_path):
    """0 never reaches eval. It is not offered by the interface (min is 10) but
    can still arrive from a config saved before the field had a minimum."""
    parameters = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path, diameter=0)

    assert parameters['eval_parameters'] == {}


def test_negative_diameter_keeps_native_resolution(tmp_path):
    """A negative diameter is meaningless; treat it as off rather than rescaling.

    cellpose itself guards with ``diameter > 0``, so this only keeps the worker
    from passing through a value that would be silently ignored downstream.
    """
    parameters = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path, diameter=-5)

    assert parameters['eval_parameters'] == {}


def test_positive_diameter_is_passed_to_eval(tmp_path):
    """A positive diameter reaches ``CellposeModel.eval()`` as a float."""
    parameters = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path, diameter=60)

    assert parameters['eval_parameters'] == {'diameter': 60.0}
    assert isinstance(parameters['eval_parameters']['diameter'], float)


def test_diameter_applies_to_custom_models_too(tmp_path):
    """Custom Girder models take the same rescaling path as the built-ins."""
    parameters = models_config.build_cellpose_parameters(
        'my custom model', tmp_path, diameter=45)

    assert parameters == {
        'model_parameters': {
            'gpu': True,
            'pretrained_model': str(tmp_path / 'my custom model'),
        },
        'eval_parameters': {'diameter': 45.0},
    }


def test_diameter_does_not_disturb_model_parameters(tmp_path):
    """Setting a diameter must not change which checkpoint is loaded."""
    without = models_config.build_cellpose_parameters('cellpose-sam', tmp_path)
    with_diameter = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path, diameter=30)

    assert without['model_parameters'] == with_diameter['model_parameters']


def test_diameter_rescale_matches_cellpose_formula():
    """The worker's preflight estimate must match cellpose's own 30/diameter."""
    assert models_config.diameter_rescale(30) == 1.0
    assert models_config.diameter_rescale(10) == 3.0
    assert models_config.diameter_rescale(60) == 0.5


def test_diameter_rescale_is_one_when_off():
    """'Off' values mean the image is handed to the net unscaled."""
    assert models_config.diameter_rescale(None) == 1.0
    assert models_config.diameter_rescale(0) == 1.0
    assert models_config.diameter_rescale(-5) == 1.0


# --- 30 px is the identity value ---------------------------------------------
#
# cellpose hardcodes ``rescale = 30. / diameter`` (models.py:269) -- the 30 is a
# literal, not a per-model ``diam_mean`` (v4 ignores that entirely). So a
# Diameter of 30 yields rescale == 1.0, exactly what ``diameter=None`` yields at
# models.py:267, and every downstream branch keys off ``rescale != 1.0``. That
# makes 30 a real, in-range "no rescaling" value, which is why it is the
# interface default -- no out-of-band 0 sentinel needed.


def test_default_diameter_is_the_native_identity_value():
    """The interface default must be cellpose's identity diameter."""
    assert models_config.DEFAULT_DIAMETER == 30.0
    assert models_config.diameter_rescale(models_config.DEFAULT_DIAMETER) == 1.0


def test_default_diameter_is_not_passed_to_eval(tmp_path):
    """At the default, nothing is forwarded -- identical to the pre-restore call."""
    parameters = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path, diameter=models_config.DEFAULT_DIAMETER)

    assert parameters['eval_parameters'] == {}


def test_diameter_thirty_is_normalized_away_for_custom_models_too(tmp_path):
    """The normalization is not special-cased to the built-in checkpoints."""
    parameters = models_config.build_cellpose_parameters(
        'my custom model', tmp_path, diameter=30)

    assert parameters['eval_parameters'] == {}


def test_values_either_side_of_thirty_still_rescale(tmp_path):
    """Only the exact identity is dropped; neighbouring values must pass through."""
    below = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path, diameter=29)
    above = models_config.build_cellpose_parameters(
        'cellpose-sam', tmp_path, diameter=31)

    assert below['eval_parameters'] == {'diameter': 29.0}
    assert above['eval_parameters'] == {'diameter': 31.0}


def test_minimum_diameter_caps_the_upscale_factor():
    """The interface minimum bounds how far a tile can be enlarged."""
    assert models_config.MIN_DIAMETER == 10.0
    assert models_config.diameter_rescale(models_config.MIN_DIAMETER) == 3.0


def test_default_diameter_is_within_the_offered_range():
    """The default must be selectable in the interface it ships with."""
    assert models_config.MIN_DIAMETER <= models_config.DEFAULT_DIAMETER
