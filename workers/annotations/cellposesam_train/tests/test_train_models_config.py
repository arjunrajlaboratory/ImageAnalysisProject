"""Unit tests for the built-in Cellpose-SAM retraining model mapping.

These exercise ``models_config`` in isolation — it must stay free of heavy
imports (cellpose/annotation_client) so it runs in the lightweight local venv
without the full worker stack. Run with:

    .cache/testvenv/bin/pytest workers/annotations/cellposesam_train/tests -q
"""

import importlib.util
from pathlib import Path


# Load under a worker-specific module name so this suite can be collected in
# the same pytest process as cellposesam's models_config tests.
_MODELS_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'models_config.py'
_SPEC = importlib.util.spec_from_file_location(
    'cellposesam_train_models_config', _MODELS_CONFIG_PATH)
models_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(models_config)


def test_default_model_resolves_to_cpsam_v2():
    """The default dropdown selection must fine-tune from the cpsam_v2 checkpoint."""
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
    """A custom model named exactly like a base label is dropped, not duplicated."""
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


def test_validate_output_model_name_normalizes_custom_name():
    assert models_config.validate_output_model_name('  my custom model  ') == 'my custom model'


def test_validate_output_model_name_rejects_empty_name():
    try:
        models_config.validate_output_model_name('   ')
    except ValueError as exc:
        assert 'provide a name' in str(exc)
    else:
        raise AssertionError('Expected an empty output name to be rejected')


def test_validate_output_model_name_rejects_builtin_labels():
    for model_name in models_config.BASE_MODELS:
        try:
            models_config.validate_output_model_name(model_name)
        except ValueError as exc:
            assert 'reserved' in str(exc)
        else:
            raise AssertionError(f'Expected {model_name!r} to be rejected')


def test_validate_output_model_name_rejects_path_like_names():
    invalid_names = [
        '.',
        '..',
        '../escaped',
        'nested/model',
        '/tmp/escaped',
    ]

    for model_name in invalid_names:
        try:
            models_config.validate_output_model_name(model_name)
        except ValueError as exc:
            assert 'plain file name' in str(exc)
        else:
            raise AssertionError(f'Expected {model_name!r} to be rejected')
