import pytest
from unittest.mock import patch, MagicMock
import json
import numpy as np

# Import your worker module
from entrypoint import compute, interface, preview


@pytest.fixture
def mock_tile_client():
    """Mock the tiles.UPennContrastDataset"""
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client:
        client = mock_client.return_value
        # Set up default tile info with multiple frames and channels
        client.tiles = {
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 1},
            ],
            'IndexRange': {
                'IndexXY': 1,
                'IndexZ': 1,
                'IndexT': 2,
                'IndexC': 2
            },
            'channels': ['DAPI', 'FITC'],
            'mm_x': 0.65,
            'mm_y': 0.65,
            'magnification': 20,
            'dtype': np.uint16
        }

        # Mock getRegion to return dummy image data
        client.getRegion.return_value = np.random.randint(0, 1000, (512, 512, 1), dtype=np.uint16)

        # Mock coordinatesToFrameIndex
        client.coordinatesToFrameIndex.return_value = 0

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
def mock_large_image():
    """Mock large_image operations"""
    with patch('large_image.new') as mock_li_new:
        mock_sink = MagicMock()
        mock_li_new.return_value = mock_sink
        yield mock_sink


@pytest.fixture
def mock_gaussian_filter():
    """Mock the skimage.filters.gaussian function"""
    with patch('skimage.filters.gaussian') as mock_gaussian:
        # Return the input image multiplied by 0.8 to simulate blur effect
        mock_gaussian.side_effect = lambda img, sigma: img * 0.8
        yield mock_gaussian


@pytest.fixture
def sample_params_basic():
    """Create basic sample parameters"""
    return {
        'workerInterface': {
            'Sigma': 2.0,
            'Channel': 0,
            'All channels': {'0': True, '1': False}
        },
        'tile': {
            'XY': 0,
            'Z': 0,
            'Time': 0
        },
        'channel': 0
    }


@pytest.fixture
def sample_params_multi_channel():
    """Create sample parameters with multiple channels"""
    return {
        'workerInterface': {
            'Sigma': 5.0,
            'Channel': 0,
            'All channels': {'0': True, '1': True}
        },
        'tile': {
            'XY': 0,
            'Z': 0,
            'Time': 0
        },
        'channel': 0
    }


@pytest.fixture
def sample_params_high_sigma():
    """Create sample parameters with high sigma value"""
    return {
        'workerInterface': {
            'Sigma': 50.0,
            'Channel': 1,
            'All channels': {'0': False, '1': True}
        },
        'tile': {
            'XY': 0,
            'Z': 0,
            'Time': 0
        },
        'channel': 1
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
    assert 'Sigma' in interface_data
    assert 'Channel' in interface_data
    assert 'All channels' in interface_data

    # Check Sigma interface
    sigma_interface = interface_data['Sigma']
    assert sigma_interface['type'] == 'number'
    assert sigma_interface['min'] == 0
    assert sigma_interface['max'] == 100
    assert sigma_interface['default'] == 20
    assert sigma_interface['displayOrder'] == 0
    assert 'tooltip' in sigma_interface

    # Check Channel interface
    channel_interface = interface_data['Channel']
    assert channel_interface['type'] == 'channel'
    assert channel_interface['default'] == 0
    assert channel_interface['displayOrder'] == 1

    # Check All channels interface
    all_channels_interface = interface_data['All channels']
    assert all_channels_interface['type'] == 'channelCheckboxes'
    assert all_channels_interface['displayOrder'] == 2


def test_compute_basic_functionality(mock_tile_client, mock_large_image, mock_gaussian_filter, sample_params_basic):
    """Test basic compute functionality with single channel"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify gaussian filter was called
    mock_gaussian_filter.assert_called()

    # Check that sigma was passed correctly
    call_args = mock_gaussian_filter.call_args_list[0]
    assert call_args[1]['sigma'] == 2.0

    # Verify sink operations
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')

    # Verify file upload
    mock_tile_client.client.uploadFileToFolder.assert_called_once_with(
        'test_dataset', '/tmp/output.tiff'
    )

    # Verify metadata was added
    expected_metadata = {
        'tool': 'Gaussian blur',
        'sigma': 2.0,
        'channel': 0
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_multi_channel(mock_tile_client, mock_large_image, mock_gaussian_filter, sample_params_multi_channel):
    """Test compute with multiple channels selected"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_multi_channel)

    # Should process frames for both channels (frames with IndexC=0 and IndexC=1)
    assert mock_gaussian_filter.call_count == 4  # 2 time points * 2 channels

    # Verify sink operations
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')

    # Verify metadata includes sigma value
    expected_metadata = {
        'tool': 'Gaussian blur',
        'sigma': 5.0,
        'channel': 0
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_high_sigma_value(mock_tile_client, mock_large_image, mock_gaussian_filter, sample_params_high_sigma):
    """Test compute with high sigma value"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_high_sigma)

    # Verify gaussian filter was called with high sigma
    call_args = mock_gaussian_filter.call_args_list[0]
    assert call_args[1]['sigma'] == 50.0

    # Verify metadata includes correct sigma and channel
    expected_metadata = {
        'tool': 'Gaussian blur',
        'sigma': 50.0,
        'channel': 1
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_no_frames_single_frame(mock_tile_client, mock_large_image, mock_gaussian_filter, sample_params_basic):
    """Test compute with single frame (no 'frames' key in tiles)"""
    # Remove frames to simulate single frame scenario
    del mock_tile_client.tiles['frames']

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Should process single frame
    mock_gaussian_filter.assert_called_once()
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')


def test_channel_filtering_logic(mock_tile_client, mock_large_image, mock_gaussian_filter):
    """Test that only selected channels are processed"""
    params = {
        'workerInterface': {
            'Sigma': 3.0,
            'Channel': 0,
            'All channels': {'0': True, '1': False}  # Only channel 0
        },
        'tile': {
            'XY': 0,
            'Z': 0,
            'Time': 0
        },
        'channel': 0
    }

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Should only process frames with IndexC=0 (2 frames total)
    assert mock_gaussian_filter.call_count == 2
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')


def test_dtype_handling_integer(mock_tile_client, mock_large_image, mock_gaussian_filter, sample_params_basic):
    """Test proper handling of integer dtypes"""
    # Set integer dtype
    mock_tile_client.tiles['dtype'] = np.uint8

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Should handle integer scaling correctly
    mock_gaussian_filter.assert_called()
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')


def test_dtype_handling_float(mock_tile_client, mock_large_image, mock_gaussian_filter, sample_params_basic):
    """Test proper handling of float dtypes"""
    # Set float dtype
    mock_tile_client.tiles['dtype'] = np.float32

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Should handle float scaling correctly
    mock_gaussian_filter.assert_called()
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')


def test_metadata_preservation(mock_tile_client, mock_large_image, sample_params_basic):
    """Test that image metadata is preserved in output"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify channel names were set on sink
    assert mock_large_image.channelNames == ['DAPI', 'FITC']

    # Verify other metadata was preserved
    assert mock_large_image.mm_x == 0.65
    assert mock_large_image.mm_y == 0.65
    assert mock_large_image.magnification == 20


def test_progress_reporting(mock_tile_client, mock_large_image, sample_params_basic, capsys):
    """Test that progress is reported during processing"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Capture stdout to verify progress was sent
    captured = capsys.readouterr()
    assert '"progress":' in captured.out
    assert '"title": "Gaussian blur"' in captured.out
    assert 'Processing frame' in captured.out


def test_preview_functionality(mock_tile_client, mock_worker_preview_client):
    """Test the preview generation functionality"""
    params = {
        'assignment': {},
        'channel': 0,
        'connectTo': {},
        'tags': [],
        'tile': {
            'XY': 0,
            'Z': 0,
            'Time': 0
        },
        'workerInterface': {
            'Sigma': 10.0
        }
    }

    with patch('imageio.imwrite') as mock_imwrite, \
            patch('base64.b64encode') as mock_b64encode:

        # Mock image writing and base64 encoding
        mock_imwrite.return_value = b'fake_png_data'
        mock_b64encode.return_value = b'fake_base64_data'

        # Run preview
        preview('test_dataset', 'http://test-api', 'test-token', params, 'test_image')

        # Verify preview was set
        mock_worker_preview_client.setWorkerImagePreview.assert_called_once()

        # Get the preview data
        call_args = mock_worker_preview_client.setWorkerImagePreview.call_args
        image_arg = call_args[0][0]
        preview_data = call_args[0][1]

        # Verify preview structure
        assert image_arg == 'test_image'
        assert 'image' in preview_data
        assert preview_data['image'].startswith('data:image/png;base64,')


def test_sigma_parameter_validation():
    """Test that sigma parameter is properly converted to float"""
    params_str = {
        'workerInterface': {
            'Sigma': '15.5',  # String value
            'Channel': 0,
            'All channels': {'0': True}
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client, \
            patch('large_image.new'), \
            patch('skimage.filters.gaussian') as mock_gaussian:

        client = mock_client.return_value
        client.tiles = {
            'frames': [{'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0}],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40,
            'dtype': np.uint8
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # This should not raise an error despite string input
        compute('test_dataset', 'http://test-api', 'test-token', params_str)

        # Verify sigma was converted to float
        call_args = mock_gaussian.call_args_list[0]
        assert call_args[1]['sigma'] == 15.5
        assert isinstance(call_args[1]['sigma'], float)


def test_channel_selection_edge_cases(mock_tile_client, mock_large_image):
    """Test channel selection with edge cases"""
    # Test with no channels selected
    params_no_channels = {
        'workerInterface': {
            'Sigma': 2.0,
            'Channel': 0,
            'All channels': {'0': False, '1': False}  # No channels
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('skimage.filters.gaussian') as mock_gaussian:
        # Run compute
        compute('test_dataset', 'http://test-api', 'test-token', params_no_channels)

        # Should not call gaussian filter since no channels are selected
        assert mock_gaussian.call_count == 0
        mock_large_image.write.assert_called_once_with('/tmp/output.tiff')


def test_coordinate_to_frame_conversion(mock_tile_client, mock_large_image, sample_params_basic):
    """Test that coordinates are properly converted to frame index"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify coordinatesToFrameIndex was called with correct parameters
    mock_tile_client.coordinatesToFrameIndex.assert_called_once_with(
        0, 0, 0, 0  # XY, Z, Time, channel
    )


def test_different_image_dimensions(mock_tile_client, mock_large_image, sample_params_basic):
    """Test handling of different image dimensions"""
    # Mock getRegion to return different sized images
    mock_tile_client.getRegion.return_value = np.random.randint(
        0, 1000, (256, 256, 1), dtype=np.uint16)

    with patch('skimage.filters.gaussian') as mock_gaussian:
        mock_gaussian.side_effect = lambda img, sigma: img * 0.8

        # Run compute
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

        # Should handle different dimensions correctly
        mock_gaussian.assert_called()
        mock_large_image.write.assert_called_once_with('/tmp/output.tiff')
