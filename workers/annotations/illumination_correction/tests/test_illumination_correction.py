import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Import your worker module
from entrypoint import compute, interface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _get_region_side_effect(*args, **kwargs):
    """Deterministic fake image per frame index: value = (frame_index + 1) * 10."""
    frame = kwargs.get('frame')
    if frame is None and len(args) > 1:
        frame = args[1]
    if frame is None:
        frame = 0
    return np.full((16, 16), (frame + 1) * 10, dtype=np.uint16)


@pytest.fixture
def mock_tile_client():
    """Mock the tiles.UPennContrastDataset"""
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client:
        client = mock_client.return_value
        # 6 frames: 2 channels x 2 XY x (mostly) 1 Z, with some time variation.
        client.tiles = {
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},  # 0
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},  # 1
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0},  # 2
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 1},  # 3
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},  # 4
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},  # 5
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

        client.getRegion.side_effect = _get_region_side_effect
        client.coordinatesToFrameIndex.return_value = 0

        mock_gc = MagicMock()
        mock_gc.uploadFileToFolder.return_value = {'itemId': 'test_item_id'}
        client.client = mock_gc

        yield client


@pytest.fixture
def mock_worker_preview_client():
    """Mock the UPennContrastWorkerPreviewClient"""
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_client:
        yield mock_client.return_value


@pytest.fixture
def mock_large_image():
    """Mock large_image operations"""
    with patch('large_image.new') as mock_li_new:
        mock_sink = MagicMock()
        mock_li_new.return_value = mock_sink
        yield mock_sink


@pytest.fixture
def mock_corrections(mocker):
    """Patch every per-algorithm function with an identity pass-through mock."""
    identity = lambda stack, opts: (np.asarray(stack, dtype=np.float64), {'note': 'mock'})
    return {
        'basic': mocker.patch('entrypoint.correct_basic', side_effect=identity),
        'cidre': mocker.patch('entrypoint.correct_cidre', side_effect=identity),
        'cellprofiler': mocker.patch('entrypoint.correct_cellprofiler', side_effect=identity),
        'flatfield': mocker.patch('entrypoint.correct_flatfield', side_effect=identity),
        'destripe': mocker.patch('entrypoint.correct_destripe', side_effect=identity),
        'sscor': mocker.patch('entrypoint.correct_sscor', side_effect=identity),
    }


def base_worker_interface(**overrides):
    interface_dict = {
        'Method': 'basic',
        'Channels to correct': {'0': True, '1': False},
        'Estimate darkfield': True,
        'Flatfield smoothness': 1.0,
        'Darkfield smoothness': 1.0,
        'Correct timelapse baseline drift': False,
        'Smoothing sigma': 50,
        'Dark quantile': 0.02,
        'CellProfiler mode': 'regular',
        'Flat-field XY coordinate': '1',
        'Dark-field XY coordinate': '',
        'Dark-field constant': 0,
        'Destripe sigma': 128,
        'Destripe wavelet': 'db3',
        'Destripe level': 0,
        'SSCOR mode': 'pretrained',
        'SSCOR stripe direction': 'horizontal',
        'SSCOR horizontal stripe count': 1,
        'SSCOR vertical stripe count': 1,
        'SSCOR grid direction': 0,
        'SSCOR training epochs': 30,
        'SSCOR patch size': 256,
        'SSCOR offset size': 100,
        'SSCOR repeat': 1,
        'SSCOR dark threshold': 10,
        'Report correction quality (QC)': False,
    }
    interface_dict.update(overrides)
    return {'workerInterface': interface_dict}


# ---------------------------------------------------------------------------
# 1. interface()
# ---------------------------------------------------------------------------

def test_interface(mock_worker_preview_client):
    interface('test_image', 'http://test-api', 'test-token')

    mock_worker_preview_client.setWorkerImageInterface.assert_called_once()

    call_args = mock_worker_preview_client.setWorkerImageInterface.call_args
    image_arg = call_args[0][0]
    interface_data = call_args[0][1]

    assert image_arg == 'test_image'

    # Method select
    assert 'Method' in interface_data
    method_iface = interface_data['Method']
    assert method_iface['type'] == 'select'
    assert method_iface['items'] == [
        'basic', 'cidre', 'cellprofiler', 'flatfield', 'destripe', 'sscor']
    assert method_iface['default'] == 'basic'

    # Channel field
    assert interface_data['Channels to correct']['type'] == 'channelCheckboxes'

    # Key params for each method are present
    for key in [
        'Estimate darkfield', 'Flatfield smoothness', 'Darkfield smoothness',
        'Correct timelapse baseline drift', 'Smoothing sigma', 'Dark quantile',
        'CellProfiler mode', 'Flat-field XY coordinate', 'Dark-field XY coordinate',
        'Dark-field constant', 'Destripe sigma', 'Destripe wavelet', 'Destripe level',
        'SSCOR mode', 'SSCOR stripe direction', 'SSCOR horizontal stripe count',
        'SSCOR vertical stripe count', 'SSCOR grid direction', 'SSCOR training epochs',
        'SSCOR patch size', 'SSCOR offset size', 'SSCOR repeat', 'SSCOR dark threshold',
        'Report correction quality (QC)',
    ]:
        assert key in interface_data, f'{key} missing from interface'

    assert interface_data['CellProfiler mode']['items'] == ['regular', 'background']
    assert interface_data['Destripe wavelet']['items'] == ['db3', 'db5', 'haar', 'sym4']
    assert interface_data['Estimate darkfield']['default'] is True
    assert interface_data['Report correction quality (QC)']['default'] is False

    assert interface_data['SSCOR mode']['type'] == 'select'
    assert interface_data['SSCOR mode']['items'] == ['pretrained', 'self-train']
    assert interface_data['SSCOR mode']['default'] == 'pretrained'
    assert interface_data['SSCOR stripe direction']['items'] == ['horizontal', 'vertical', 'grid']
    assert interface_data['SSCOR stripe direction']['default'] == 'horizontal'
    assert interface_data['SSCOR horizontal stripe count']['default'] == 1
    assert interface_data['SSCOR vertical stripe count']['default'] == 1
    assert interface_data['SSCOR grid direction']['default'] == 0
    assert interface_data['SSCOR training epochs']['default'] == 30


# ---------------------------------------------------------------------------
# 2. Dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['basic', 'cidre', 'cellprofiler', 'flatfield', 'destripe'])
def test_dispatch_calls_only_selected_method(
        method, mock_tile_client, mock_large_image, mock_corrections):
    params = base_worker_interface(Method=method)

    compute('test_dataset', 'http://test-api', 'test-token', params)

    for name, mock_fn in mock_corrections.items():
        if name == method:
            assert mock_fn.call_count == 1, f'{name} should have been called once'
        else:
            assert mock_fn.call_count == 0, f'{name} should not have been called'


# ---------------------------------------------------------------------------
# 3. Channel filtering
# ---------------------------------------------------------------------------

def test_unselected_channels_pass_through_unchanged(
        mock_tile_client, mock_large_image, mock_corrections):
    offset_correction = lambda stack, opts: (np.asarray(stack, dtype=np.float64) + 1000, {})
    mock_corrections['basic'].side_effect = offset_correction

    params = base_worker_interface(Method='basic', **{'Channels to correct': {'0': True, '1': False}})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Only channel 0's collection (3 frames) should have gone through correct_basic
    assert mock_corrections['basic'].call_count == 1
    stack_arg = mock_corrections['basic'].call_args[0][0]
    assert stack_arg.shape[0] == 3

    add_tile_calls = mock_large_image.addTile.call_args_list
    for c in add_tile_calls:
        image = c.args[0]
        channel = c.kwargs.get('c')
        frame_idx = _frame_index_from_kwargs(mock_tile_client, c.kwargs)
        expected_raw = (frame_idx + 1) * 10
        if channel == 0:
            # Processed: raw + 1000 (clipped/cast back to uint16)
            assert int(image.flat[0]) == expected_raw + 1000
        else:
            # Untouched: exactly the raw mock image
            assert int(image.flat[0]) == expected_raw


def _frame_index_from_kwargs(tile_client, kwargs):
    """Recover the original frame index (0..5) from the addTile xy/z/t/c kwargs."""
    for i, frame in enumerate(tile_client.tiles['frames']):
        if (frame['IndexXY'] == kwargs.get('xy') and frame['IndexZ'] == kwargs.get('z')
                and frame['IndexT'] == kwargs.get('t') and frame['IndexC'] == kwargs.get('c')):
            return i
    raise AssertionError(f'Could not match frame for kwargs {kwargs}')


# ---------------------------------------------------------------------------
# 4. Output plumbing
# ---------------------------------------------------------------------------

def test_output_plumbing(mock_tile_client, mock_large_image, mock_corrections):
    params = base_worker_interface(Method='basic')
    compute('test_dataset', 'http://test-api', 'test-token', params)

    mock_large_image.write.assert_called_once_with('/tmp/illumination_corrected.tiff')
    mock_tile_client.client.uploadFileToFolder.assert_called_once_with(
        'test_dataset', '/tmp/illumination_corrected.tiff')

    mock_tile_client.client.addMetadataToItem.assert_called_once()
    call_args = mock_tile_client.client.addMetadataToItem.call_args
    assert call_args[0][0] == 'test_item_id'
    metadata = call_args[0][1]
    assert metadata['tool'] == 'Illumination Correction'
    assert metadata['method'] == 'basic'
    assert metadata['channels'] == [0]


# ---------------------------------------------------------------------------
# 5. Metadata preservation
# ---------------------------------------------------------------------------

def test_metadata_preservation(mock_tile_client, mock_large_image, mock_corrections):
    params = base_worker_interface(Method='basic')
    compute('test_dataset', 'http://test-api', 'test-token', params)

    assert mock_large_image.channelNames == ['DAPI', 'FITC']
    assert mock_large_image.mm_x == 0.65
    assert mock_large_image.mm_y == 0.65
    assert mock_large_image.magnification == 20


# ---------------------------------------------------------------------------
# 6. Error paths
# ---------------------------------------------------------------------------

def test_no_channels_selected_error(mock_tile_client, mock_large_image, mock_corrections, capsys):
    params = base_worker_interface(**{'Channels to correct': {'0': False, '1': False}})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"error": "No channels to correct"' in captured.out
    assert '"type": "error"' in captured.out
    mock_large_image.write.assert_not_called()


def test_no_frames_error(mock_tile_client, mock_large_image, mock_corrections, capsys):
    del mock_tile_client.tiles['frames']

    params = base_worker_interface()
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"error": "Only one image; exiting"' in captured.out
    assert '"type": "error"' in captured.out
    mock_large_image.write.assert_not_called()


def test_flatfield_without_flat_reference_error(
        mock_tile_client, mock_large_image, mock_corrections, capsys):
    params = base_worker_interface(Method='flatfield', **{'Flat-field XY coordinate': ''})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"error": "Flat-field reference required"' in captured.out
    mock_corrections['flatfield'].assert_not_called()
    mock_large_image.write.assert_not_called()


def test_sscor_dispatch_with_weights(
        mock_tile_client, mock_large_image, mock_corrections, mocker):
    """With SSCOR_WEIGHTS resolvable, sscor dispatches to correct_sscor and writes output."""
    mocker.patch('entrypoint.resolve_sscor_checkpoint',
                  return_value='/fake/weights/latest_net_G.pth')

    params = base_worker_interface(Method='sscor')
    compute('test_dataset', 'http://test-api', 'test-token', params)

    mock_corrections['sscor'].assert_called_once()
    call_opts = mock_corrections['sscor'].call_args[0][1]
    assert call_opts['sscor_weights'] == '/fake/weights/latest_net_G.pth'
    assert call_opts['sscor_gpu_ids'] in ('0', '-1')
    assert call_opts['sscor_patch_size'] == 256
    assert call_opts['sscor_offset_size'] == 100
    assert call_opts['sscor_repeat'] == 1
    assert call_opts['sscor_dark_threshold'] == 10

    mock_large_image.write.assert_called_once_with('/tmp/illumination_corrected.tiff')
    mock_tile_client.client.uploadFileToFolder.assert_called_once()


def test_sscor_without_weights_error(
        mock_tile_client, mock_large_image, mock_corrections, mocker, capsys):
    """With no SSCOR_WEIGHTS (simulated via resolve_sscor_checkpoint -> None, which already
    sent the actionable sendError), compute() must bail out before the channel loop: no
    correct_sscor call, no write, no upload."""
    mocker.patch('entrypoint.resolve_sscor_checkpoint', return_value=None)

    params = base_worker_interface(Method='sscor')
    compute('test_dataset', 'http://test-api', 'test-token', params)

    mock_corrections['sscor'].assert_not_called()
    mock_large_image.write.assert_not_called()
    mock_tile_client.client.uploadFileToFolder.assert_not_called()


def test_sscor_selftrain_no_weights_needed(
        mock_tile_client, mock_large_image, mock_corrections, mocker, capsys):
    """'SSCOR mode'='self-train' must dispatch to correct_sscor with no SSCOR_WEIGHTS at all
    and WITHOUT resolve_sscor_checkpoint being patched -- self-train trains its own model per
    frame instead of requiring a pre-supplied checkpoint."""
    mocker.patch.dict('os.environ', {'SSCOR_WEIGHTS': ''})

    resolve_spy = mocker.patch('entrypoint.resolve_sscor_checkpoint')

    params = base_worker_interface(Method='sscor', **{'SSCOR mode': 'self-train'})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"error"' not in captured.out
    resolve_spy.assert_not_called()

    mock_corrections['sscor'].assert_called_once()
    call_opts = mock_corrections['sscor'].call_args[0][1]
    assert call_opts['sscor_mode'] == 'self-train'
    assert call_opts['sscor_weights'] is None
    assert call_opts['sscor_gpu_ids'] in ('0', '-1')
    assert call_opts['sscor_stripe_direction'] == 'horizontal'
    assert call_opts['sscor_h_n'] == 1
    assert call_opts['sscor_v_n'] == 1
    assert call_opts['sscor_grid_direction'] == 0
    assert call_opts['sscor_epochs'] == 30

    mock_large_image.write.assert_called_once_with('/tmp/illumination_corrected.tiff')
    mock_tile_client.client.uploadFileToFolder.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Progress reporting
# ---------------------------------------------------------------------------

def test_progress_reporting(mock_tile_client, mock_large_image, mock_corrections, capsys):
    params = base_worker_interface(Method='basic')
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"progress":' in captured.out
    assert 'Illumination correction' in captured.out


# ---------------------------------------------------------------------------
# Extra coverage: dtype preservation, QC metrics, small-collection warning
# ---------------------------------------------------------------------------

def test_dtype_preserved_on_output(mock_tile_client, mock_large_image, mock_corrections):
    params = base_worker_interface(Method='basic')
    compute('test_dataset', 'http://test-api', 'test-token', params)

    for c in mock_large_image.addTile.call_args_list:
        image = c.args[0]
        assert image.dtype == np.uint16


def test_qc_metrics_added_to_metadata(mock_tile_client, mock_large_image, mock_corrections):
    params = base_worker_interface(Method='basic', **{'Report correction quality (QC)': True})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    metadata = mock_tile_client.client.addMetadataToItem.call_args[0][1]
    assert 'qc' in metadata
    assert 0 in metadata['qc']
    qc = metadata['qc'][0]
    assert 'cv_mean_image' in qc
    assert 'corner_center_ratio' in qc
    assert 'interframe_cv' in qc


def test_small_collection_warning(mock_tile_client, mock_large_image, mock_corrections, capsys):
    # Reduce the dataset so channel 0 only has a single frame.
    mock_tile_client.tiles['frames'] = [
        {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
        {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
    ]

    params = base_worker_interface(Method='basic')
    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert '"warning": "Small image collection"' in captured.out


def test_flatfield_dispatch_uses_reference_frames(
        mock_tile_client, mock_large_image, mock_corrections):
    params = base_worker_interface(
        Method='flatfield',
        **{'Flat-field XY coordinate': '1', 'Dark-field XY coordinate': ''})
    compute('test_dataset', 'http://test-api', 'test-token', params)

    mock_corrections['flatfield'].assert_called_once()
    call_opts = mock_corrections['flatfield'].call_args[0][1]
    assert 'flat' in call_opts
    assert 'dark' in call_opts
    assert call_opts['flat'].shape == (16, 16)
    assert call_opts['dark'].shape == (16, 16)
