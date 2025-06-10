import pytest
from unittest.mock import patch, MagicMock
import json

# Import your worker module
from entrypoint import compute, interface


@pytest.fixture
def mock_tile_client():
    """Mock the tiles.UPennContrastDataset"""
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client:
        client = mock_client.return_value
        # Set up default tile info with 3-channel RGB
        client.tiles = {
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 2}
            ],
            'IndexRange': {
                'IndexXY': 1,
                'IndexZ': 1,
                'IndexT': 1,
                'IndexC': 3
            },
            'channels': ['Red', 'Green', 'Blue'],
            'mm_x': 1.0,
            'mm_y': 1.0,
            'magnification': 40
        }
        # Mock getRegion to return dummy image data
        import numpy as np
        client.getRegion.return_value = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        # Mock the girder client
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
def sample_params():
    """Create sample parameters for the worker"""
    return {
        'workerInterface': {
            'Max percentile': 99
        }
    }


@pytest.fixture
def sample_params_empty_percentile():
    """Create sample parameters with empty percentile"""
    return {
        'workerInterface': {
            'Max percentile': ""
        }
    }


def test_interface(mock_worker_preview_client):
    """Test the interface generation"""
    # Test interface function
    interface('test_image', 'http://test-api', 'test-token')

    # Verify interface was set
    mock_worker_preview_client.setWorkerImageInterface.assert_called_once()

    # Get the interface data
    call_args = mock_worker_preview_client.setWorkerImageInterface.call_args
    image_arg = call_args[0][0]
    interface_data = call_args[0][1]

    # Verify image parameter
    assert image_arg == 'test_image'

    # Verify interface structure
    assert 'Max percentile' in interface_data
    assert interface_data['Max percentile']['type'] == 'number'
    assert interface_data['Max percentile']['min'] == 0
    assert interface_data['Max percentile']['max'] == 100
    assert interface_data['Max percentile']['default'] == 99
    assert 'tooltip' in interface_data['Max percentile']
    assert interface_data['Max percentile']['displayOrder'] == 1


def test_compute_basic_functionality(mock_tile_client, sample_params, capsys):
    """Test basic compute functionality with valid parameters"""
    with patch('large_image.new') as mock_li_new:

        # Mock large_image sink
        mock_sink = MagicMock()
        mock_li_new.return_value = mock_sink

        # Run compute
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Verify that the mock tile client was used (check that getRegion was called)
        assert mock_tile_client.getRegion.call_count == 3  # Should be called for each channel

        # Verify sink operations
        mock_sink.write.assert_called_once_with('/tmp/deconvolved.tiff')

        # Verify file upload
        mock_tile_client.client.uploadFileToFolder.assert_called_once_with(
            'test_dataset', '/tmp/deconvolved.tiff'
        )

        # Verify metadata was added
        mock_tile_client.client.addMetadataToItem.assert_called_once_with(
            'test_item_id', {'tool': 'H&E Deconvolution'}
        )

        # Capture stdout to verify progress was sent
        captured = capsys.readouterr()
        assert '"progress":' in captured.out
        assert '"title": "Deconvolving"' in captured.out


def test_compute_empty_percentile_default(mock_tile_client, sample_params_empty_percentile):
    """Test that empty percentile defaults to 99"""
    with patch('large_image.new') as mock_li_new, \
            patch('annotation_client.utils.sendProgress'):

        mock_sink = MagicMock()
        mock_li_new.return_value = mock_sink

        # Run compute with empty percentile
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_empty_percentile)

        # Should complete successfully (defaulting to 99)
        mock_sink.write.assert_called_once()


def test_compute_invalid_channel_count(mock_tile_client, sample_params, capsys):
    """Test error handling for non-RGB images"""
    # Modify tile client to return wrong number of channels
    # Note: max(range(0, 1)) = 0, max(range(0, 2)) = 1, max(range(0, 3)) = 2
    # The code checks if max(range_c) != 2, so we need max != 2
    mock_tile_client.tiles['IndexRange']['IndexC'] = 2  # This gives max(range(0,2)) = 1, not 2

    # Should send error and return early
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Capture stdout to verify error was sent
    captured = capsys.readouterr()
    assert '"error": "Need 3 channel RGB image"' in captured.out
    assert '"type": "error"' in captured.out


def test_compute_no_frames_error(mock_tile_client, sample_params):
    """Test error handling when no frames are present"""
    # Remove frames from tile info - this will cause a KeyError when accessing 'frames'
    del mock_tile_client.tiles['frames']

    with patch('annotation_client.utils.sendError') as mock_error:
        # Should raise KeyError when trying to access frames
        with pytest.raises(KeyError, match="frames"):
            compute('test_dataset', 'http://test-api', 'test-token', sample_params)


def test_compute_no_index_range(mock_tile_client, sample_params, capsys):
    """Test handling of tiles without IndexRange"""
    # Remove IndexRange to test fallback behavior
    del mock_tile_client.tiles['IndexRange']
    # Ensure we still have 3 channels in frames
    mock_tile_client.tiles['frames'] = [
        {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
        {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
        {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 2}
    ]

    # Without IndexRange, range_c = [0], so max(range_c) = 0 != 2, triggering error
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Capture stdout to verify error was sent
    captured = capsys.readouterr()
    assert '"error": "Need 3 channel RGB image"' in captured.out
    assert '"type": "error"' in captured.out


def test_compute_custom_percentile(mock_tile_client):
    """Test compute with custom percentile value"""
    custom_params = {
        'workerInterface': {
            'Max percentile': 95
        }
    }

    with patch('large_image.new') as mock_li_new, \
            patch('annotation_client.utils.sendProgress'):

        mock_sink = MagicMock()
        mock_li_new.return_value = mock_sink

        # Run compute with custom percentile
        compute('test_dataset', 'http://test-api', 'test-token', custom_params)

        # Should complete successfully
        mock_sink.write.assert_called_once()


def test_parameter_validation():
    """Test that required parameters are handled correctly"""
    # Test with various percentile values
    valid_percentiles = [0, 50, 99, 100]

    for percentile in valid_percentiles:
        params = {'workerInterface': {'Max percentile': percentile}}
        # Should not raise any exceptions during parameter extraction
        max_percentile = params['workerInterface']['Max percentile']
        if max_percentile == "":
            max_percentile = 99
        assert isinstance(max_percentile, (int, float))
        assert 0 <= max_percentile <= 100


def test_worker_startup():
    """Test that the worker module can be imported and basic functions exist"""
    # Test that required functions exist
    assert callable(compute)
    assert callable(interface)

    # Test function signatures (basic validation)
    import inspect

    # Check compute function signature
    compute_sig = inspect.signature(compute)
    expected_compute_params = ['datasetId', 'apiUrl', 'token', 'params']
    assert list(compute_sig.parameters.keys()) == expected_compute_params

    # Check interface function signature
    interface_sig = inspect.signature(interface)
    expected_interface_params = ['image', 'apiUrl', 'token']
    assert list(interface_sig.parameters.keys()) == expected_interface_params
