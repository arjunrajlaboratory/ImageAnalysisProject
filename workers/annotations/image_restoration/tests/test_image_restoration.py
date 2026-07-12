import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import the worker module under test. All heavy third-party imports
# (torch, careamics, cellpose, the vendored zs_deconvnet/fluoresfm code) are
# lazy inside the per-algorithm restore_* functions, so this module imports
# cleanly without any of those packages installed.
from entrypoint import (
    compute,
    interface,
    resolve_device,
    resolve_fluoresfm_weights,
    resolve_fluoresfm_embedder,
    _build_method_opts,
    _clip_to_dtype,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tile_client():
    """Mock the tiles.UPennContrastDataset with a 2-channel, multi-frame dataset."""
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client:
        client = mock_client.return_value
        client.tiles = {
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 1},
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
            ],
            'IndexRange': {
                'IndexXY': 2,
                'IndexZ': 1,
                'IndexT': 2,
                'IndexC': 2,
            },
            'channels': ['DAPI', 'FITC'],
            'mm_x': 0.65,
            'mm_y': 0.65,
            'magnification': 20,
            'dtype': np.uint16,
        }
        # Channel 0 -> frame indices 0, 2, 4 ; Channel 1 -> frame indices 1, 3, 5
        client.getRegion.return_value = np.random.randint(0, 1000, (64, 64), dtype=np.uint16)

        mock_gc = MagicMock()
        mock_gc.uploadFileToFolder.return_value = {'itemId': 'test_item_id'}
        client.client = mock_gc

        yield client


@pytest.fixture
def mock_worker_preview_client():
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_client:
        yield mock_client.return_value


@pytest.fixture
def mock_large_image():
    with patch('large_image.new') as mock_li_new:
        mock_sink = MagicMock()
        mock_li_new.return_value = mock_sink
        yield mock_sink


def _passthrough_restore(stack, opts, device):
    """Benign stand-in for a real restore_* function: identity, cast to float32."""
    return np.asarray(stack, dtype=np.float32)


@pytest.fixture
def mock_restore_all():
    """Patch all four per-algorithm functions with an identity mock.

    Used by tests that exercise the surrounding plumbing (dispatch, channel
    filtering, output/metadata, error paths, progress) without depending on
    any restoration algorithm's real numerics or third-party libraries --
    mirrors histogram_matching/tests/test_histogram_matching.py's
    `entrypoint.match_histograms` patch.
    """
    with patch('entrypoint.restore_n2v', side_effect=_passthrough_restore) as m_n2v, \
         patch('entrypoint.restore_cellpose3', side_effect=_passthrough_restore) as m_cellpose3, \
         patch('entrypoint.restore_zs_deconvnet', side_effect=_passthrough_restore) as m_zs, \
         patch('entrypoint.restore_fluoresfm', side_effect=_passthrough_restore) as m_fluoresfm:
        yield {
            'n2v': m_n2v,
            'cellpose3': m_cellpose3,
            'zs_deconvnet': m_zs,
            'fluoresfm': m_fluoresfm,
        }


def base_params(method='n2v', channels=None, use_gpu=False, **extra_interface):
    """Build a minimal params dict. `use_gpu` defaults to False so tests never
    need a real torch install (resolve_device only imports torch when GPU use
    is actually requested)."""
    if channels is None:
        channels = {'0': True, '1': False}
    worker_interface = {
        'Method': method,
        'Channels to restore': channels,
        'Use GPU': use_gpu,
    }
    worker_interface.update(extra_interface)
    return {'workerInterface': worker_interface}


# ---------------------------------------------------------------------------
# interface()
# ---------------------------------------------------------------------------

def test_interface(mock_worker_preview_client):
    interface('test_image', 'http://test-api', 'test-token')

    mock_worker_preview_client.setWorkerImageInterface.assert_called_once()
    call_args = mock_worker_preview_client.setWorkerImageInterface.call_args
    image_arg = call_args[0][0]
    interface_data = call_args[0][1]

    assert image_arg == 'test_image'

    # Method select
    method_field = interface_data['Method']
    assert method_field['type'] == 'select'
    assert method_field['items'] == ['n2v', 'cellpose3', 'zs_deconvnet', 'fluoresfm']
    assert method_field['default'] == 'n2v'

    # Channel selection
    assert interface_data['Channels to restore']['type'] == 'channelCheckboxes'

    # Use GPU
    assert interface_data['Use GPU']['type'] == 'checkbox'
    assert interface_data['Use GPU']['default'] is True

    # n2v params
    assert interface_data['Epochs']['type'] == 'number'
    assert interface_data['Epochs']['default'] == 20
    assert interface_data['Use N2V2']['type'] == 'checkbox'
    assert interface_data['Use N2V2']['default'] is True
    assert interface_data['Patch size']['default'] == 64

    # cellpose3 params
    cellpose_field = interface_data['Cellpose3 model']
    assert cellpose_field['type'] == 'select'
    assert cellpose_field['default'] == 'denoise_cyto3'
    assert 'denoise_cyto3' in cellpose_field['items']
    assert 'upsample_cyto3' in cellpose_field['items']

    # zs_deconvnet params
    assert interface_data['ZS iterations']['type'] == 'number'
    assert interface_data['ZS upsampling']['type'] == 'checkbox'
    assert interface_data['Numerical Aperture (NA)']['type'] == 'number'
    assert interface_data['Emission Wavelength (nm)']['type'] == 'number'
    assert interface_data['Pixel Size XY (nm)']['type'] == 'number'

    # fluoresfm params
    backbone_field = interface_data['FluoResFM backbone']
    assert backbone_field['type'] == 'select'
    assert backbone_field['items'] == ['unet_sd_c', 'care', 'dfcan', 'unifmir']
    assert backbone_field['default'] == 'unet_sd_c'

    task_field = interface_data['FluoResFM task']
    assert task_field['type'] == 'select'
    assert task_field['items'] == ['denoise', 'deconvolution', 'super-resolution']
    assert task_field['default'] == 'denoise'
    assert interface_data['FluoResFM text prompt']['type'] == 'text'


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['n2v', 'cellpose3', 'zs_deconvnet', 'fluoresfm'])
def test_dispatch_calls_only_selected_method(mock_tile_client, mock_large_image, mock_restore_all, method):
    params = base_params(method=method, channels={'0': True, '1': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    for name, mock_fn in mock_restore_all.items():
        if name == method:
            mock_fn.assert_called()
        else:
            mock_fn.assert_not_called()


def test_unknown_method_sends_error(mock_tile_client, mock_large_image, mock_restore_all, capsys):
    params = base_params(method='not_a_real_method', channels={'0': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"type": "error"' in captured.out
    for mock_fn in mock_restore_all.values():
        mock_fn.assert_not_called()
    mock_large_image.write.assert_not_called()


# ---------------------------------------------------------------------------
# Channel filtering
# ---------------------------------------------------------------------------

def test_channel_filtering_only_selected_processed(mock_tile_client, mock_large_image, mock_restore_all):
    params = base_params(method='n2v', channels={'0': True, '1': False})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Channel 0 has 3 frames in the fixture (indices 0, 2, 4)
    mock_restore_all['n2v'].assert_called_once()
    stack_arg = mock_restore_all['n2v'].call_args[0][0]
    assert stack_arg.shape[0] == 3

    # All 6 frames still make it into the sink (3 restored + 3 passthrough)
    assert mock_large_image.addTile.call_count == 6


def test_channel_filtering_multi_channel(mock_tile_client, mock_large_image, mock_restore_all):
    params = base_params(method='n2v', channels={'0': True, '1': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Called once per selected channel
    assert mock_restore_all['n2v'].call_count == 2


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------

def test_output_plumbing(mock_tile_client, mock_large_image, mock_restore_all):
    params = base_params(method='n2v', channels={'0': True, '1': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    mock_large_image.write.assert_called_once_with('/tmp/restored.tiff')
    mock_tile_client.client.uploadFileToFolder.assert_called_once_with(
        'test_dataset', '/tmp/restored.tiff')

    mock_tile_client.client.addMetadataToItem.assert_called_once()
    item_id, metadata = mock_tile_client.client.addMetadataToItem.call_args[0]
    assert item_id == 'test_item_id'
    assert metadata['tool'] == 'Image Restoration'
    assert metadata['method'] == 'n2v'
    assert set(metadata['channels']) == {0, 1}
    assert 'device_used' in metadata
    assert 'gpu_requested' in metadata


def test_frame_parameter_construction(mock_tile_client, mock_large_image, mock_restore_all):
    params = base_params(method='n2v', channels={'0': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    mock_large_image.addTile.assert_called()
    for call in mock_large_image.addTile.call_args_list:
        kwargs = call[1]
        assert any(key in ['xy', 'z', 't', 'c'] for key in kwargs.keys())


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

def test_metadata_preservation(mock_tile_client, mock_large_image, mock_restore_all):
    params = base_params(method='n2v', channels={'0': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    assert mock_large_image.channelNames == ['DAPI', 'FITC']
    assert mock_large_image.mm_x == 0.65
    assert mock_large_image.mm_y == 0.65
    assert mock_large_image.magnification == 20


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_no_channels_selected_error(mock_tile_client, mock_large_image, mock_restore_all, capsys):
    params = base_params(method='n2v', channels={'0': False, '1': False})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"error": "No channels selected for restoration"' in captured.out
    assert '"type": "error"' in captured.out
    mock_large_image.write.assert_not_called()
    for mock_fn in mock_restore_all.values():
        mock_fn.assert_not_called()


def test_no_frames_error(mock_tile_client, mock_large_image, mock_restore_all, capsys):
    del mock_tile_client.tiles['frames']
    params = base_params(method='n2v', channels={'0': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"error": "No frames found in dataset"' in captured.out
    mock_large_image.write.assert_not_called()


def test_restore_exception_sends_error_not_crash(mock_tile_client, mock_large_image, capsys):
    with patch('entrypoint.restore_n2v', side_effect=RuntimeError("boom")):
        params = base_params(method='n2v', channels={'0': True})
        compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"type": "error"' in captured.out
    assert 'boom' in captured.out
    mock_large_image.write.assert_not_called()


def test_single_frame_dataset_handled(mock_large_image, mock_restore_all):
    """A single-frame dataset should be processed normally, not crash."""
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client:
        client = mock_client.return_value
        client.tiles = {
            'frames': [{'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0}],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40,
            'dtype': np.uint8,
        }
        client.getRegion.return_value = np.zeros((32, 32), dtype=np.uint8)
        mock_gc = MagicMock()
        mock_gc.uploadFileToFolder.return_value = {'itemId': 'solo_item'}
        client.client = mock_gc

        params = base_params(method='n2v', channels={'0': True})
        compute('test_dataset', 'http://test-api', 'test-token', params)

        mock_large_image.write.assert_called_once_with('/tmp/restored.tiff')
        mock_gc.addMetadataToItem.assert_called_once()


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def test_progress_reporting(mock_tile_client, mock_large_image, mock_restore_all, capsys):
    params = base_params(method='n2v', channels={'0': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"progress":' in captured.out
    assert 'Complete' in captured.out


# ---------------------------------------------------------------------------
# resolve_device (GPU -> CPU fallback)
# ---------------------------------------------------------------------------

def test_resolve_device_cpu_when_gpu_not_requested():
    # Should never import torch when use_gpu is False.
    assert resolve_device(False) == 'cpu'


def test_resolve_device_gpu_available(monkeypatch):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    assert resolve_device(True) == 'cuda'


def test_resolve_device_falls_back_to_cpu_with_warning(monkeypatch, capsys):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    result = resolve_device(True)

    assert result == 'cpu'
    captured = capsys.readouterr()
    assert '"type": "warning"' in captured.out
    assert 'GPU requested but CUDA is not available' in captured.out


def test_compute_records_gpu_device_used(mock_tile_client, mock_large_image, mock_restore_all, monkeypatch):
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)

    params = base_params(method='n2v', channels={'0': True}, use_gpu=True)
    compute('test_dataset', 'http://test-api', 'test-token', params)

    metadata = mock_tile_client.client.addMetadataToItem.call_args[0][1]
    assert metadata['gpu_requested'] is True
    assert metadata['device_used'] == 'cuda'


# ---------------------------------------------------------------------------
# FluoResFM weight resolution / graceful degradation
# ---------------------------------------------------------------------------

def test_resolve_fluoresfm_weights_missing_sends_error(monkeypatch, capsys):
    monkeypatch.delenv('FLUORESFM_WEIGHTS', raising=False)

    result = resolve_fluoresfm_weights()

    assert result is None
    captured = capsys.readouterr()
    assert '"type": "error"' in captured.out
    assert 'FluoResFM weights are not available' in captured.out


def test_resolve_fluoresfm_weights_present(monkeypatch, tmp_path):
    weights_file = tmp_path / "fluoresfm.pt"
    weights_file.write_bytes(b"fake-checkpoint")
    monkeypatch.setenv('FLUORESFM_WEIGHTS', str(weights_file))

    result = resolve_fluoresfm_weights()

    assert result == str(weights_file)


def test_resolve_fluoresfm_embedder_missing_sends_error(monkeypatch, capsys):
    monkeypatch.delenv('FLUORESFM_EMBEDDER_DIR', raising=False)

    result = resolve_fluoresfm_embedder()

    assert result is None
    captured = capsys.readouterr()
    assert '"type": "error"' in captured.out
    assert 'BiomedCLIP text embedder is not available' in captured.out


def test_resolve_fluoresfm_embedder_present(monkeypatch, tmp_path):
    (tmp_path / 'open_clip_config.json').write_text('{}')
    (tmp_path / 'open_clip_pytorch_model.bin').write_bytes(b'fake')
    monkeypatch.setenv('FLUORESFM_EMBEDDER_DIR', str(tmp_path))

    result = resolve_fluoresfm_embedder()

    assert result == (
        str(tmp_path / 'open_clip_config.json'),
        str(tmp_path / 'open_clip_pytorch_model.bin'),
    )


def test_fluoresfm_missing_weights_aborts_compute_cleanly(mock_tile_client, mock_large_image, monkeypatch):
    """Simulates restore_fluoresfm() detecting missing weights (via
    resolve_fluoresfm_weights) and returning None; compute() must abort
    without writing output or crashing."""
    monkeypatch.delenv('FLUORESFM_WEIGHTS', raising=False)

    with patch('entrypoint.restore_fluoresfm', return_value=None) as mock_fluoresfm:
        params = base_params(method='fluoresfm', channels={'0': True})
        compute('test_dataset', 'http://test-api', 'test-token', params)
        mock_fluoresfm.assert_called_once()

    mock_large_image.write.assert_not_called()


# ---------------------------------------------------------------------------
# Small pure-function unit tests
# ---------------------------------------------------------------------------

def test_build_method_opts_n2v():
    opts = _build_method_opts('n2v', {'Epochs': 50, 'Use N2V2': False, 'Patch size': 32})
    assert opts == {'epochs': 50, 'use_n2v2': False, 'patch_size': 32}


def test_build_method_opts_cellpose3_default():
    opts = _build_method_opts('cellpose3', {})
    assert opts == {'model_type': 'denoise_cyto3'}


def test_build_method_opts_zs_deconvnet():
    opts = _build_method_opts('zs_deconvnet', {
        'ZS iterations': 100,
        'ZS upsampling': True,
        'Numerical Aperture (NA)': 1.2,
        'Emission Wavelength (nm)': 488,
        'Pixel Size XY (nm)': 110,
    })
    assert opts['iterations'] == 100
    assert opts['upsampling'] is True
    assert opts['NA'] == 1.2
    assert opts['wavelength'] == 488
    assert opts['pixel_size_xy'] == 110


def test_build_method_opts_fluoresfm_default_prompt():
    opts = _build_method_opts('fluoresfm', {'FluoResFM task': 'deconvolution', 'FluoResFM text prompt': ''})
    assert opts['task'] == 'deconvolution'
    assert 'deconvolution' in opts['prompt']
    # Backbone defaults to the real text-conditioned foundation model.
    assert opts['backbone'] == 'unet_sd_c'


def test_build_method_opts_fluoresfm_custom_prompt():
    opts = _build_method_opts('fluoresfm', {
        'FluoResFM task': 'denoise',
        'FluoResFM text prompt': 'a custom prompt',
    })
    assert opts['prompt'] == 'a custom prompt'


def test_build_method_opts_fluoresfm_backbone_selection():
    opts = _build_method_opts('fluoresfm', {
        'FluoResFM task': 'super-resolution',
        'FluoResFM backbone': 'dfcan',
    })
    assert opts['backbone'] == 'dfcan'
    assert opts['task'] == 'super-resolution'


def test_clip_to_dtype_uint16_clips_and_casts():
    image = np.array([-5.0, 70000.0, 100.0], dtype=np.float32)
    clipped = _clip_to_dtype(image, np.uint16)

    assert clipped.dtype == np.uint16
    assert clipped[0] == 0
    assert clipped[1] == 65535
    assert clipped[2] == 100


def test_clip_to_dtype_handles_nan_and_inf():
    image = np.array([np.nan, np.inf, -np.inf, 5.0], dtype=np.float32)
    clipped = _clip_to_dtype(image, np.uint8)

    assert not np.isnan(clipped).any()
    assert np.isfinite(clipped).all()


def test_clip_to_dtype_none_dtype_passthrough():
    image = np.array([1.5, 2.5], dtype=np.float32)
    result = _clip_to_dtype(image, None)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [1.5, 2.5])
