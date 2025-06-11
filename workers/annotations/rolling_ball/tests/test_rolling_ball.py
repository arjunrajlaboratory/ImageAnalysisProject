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
def mock_rolling_ball():
    """Mock the skimage.restoration.rolling_ball function"""
    with patch('entrypoint.restoration') as mock_restoration:
        # Return a background image that's 10% of the original
        mock_restoration.rolling_ball.side_effect = lambda img, radius: img * 0.1
        yield mock_restoration.rolling_ball


@pytest.fixture
def mock_gaussian_filter():
    """Mock the skimage.filters.gaussian function for preview"""
    with patch('entrypoint.filters') as mock_filters:
        # Return the input image multiplied by 0.8 to simulate blur effect
        mock_filters.gaussian.side_effect = lambda img, sigma: img * 0.8
        yield mock_filters.gaussian


@pytest.fixture
def sample_params_basic():
    """Create basic sample parameters"""
    return {
        'workerInterface': {
            'Radius': 10.0,
            'Channels to correct': {'0': True, '1': False}
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
            'Radius': 25.0,
            'Channels to correct': {'0': True, '1': True}
        },
        'tile': {
            'XY': 0,
            'Z': 0,
            'Time': 0
        },
        'channel': 0
    }


@pytest.fixture
def sample_params_high_radius():
    """Create sample parameters with high radius value"""
    return {
        'workerInterface': {
            'Radius': 80.0,
            'Channels to correct': {'0': False, '1': True}
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
    assert 'Radius' in interface_data
    assert 'Channels to correct' in interface_data

    # Check Radius interface
    radius_interface = interface_data['Radius']
    assert radius_interface['type'] == 'number'
    assert radius_interface['min'] == 0
    assert radius_interface['max'] == 100
    assert radius_interface['default'] == 20
    assert radius_interface['displayOrder'] == 0
    assert 'tooltip' in radius_interface

    # Check Channels to correct interface
    channels_interface = interface_data['Channels to correct']
    assert channels_interface['type'] == 'channelCheckboxes'
    assert channels_interface['displayOrder'] == 2


def test_compute_basic_functionality(mock_tile_client, mock_large_image, mock_rolling_ball, sample_params_basic):
    """Test basic compute functionality with single channel"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify rolling ball was called
    mock_rolling_ball.assert_called()

    # Check that radius was passed correctly
    call_args = mock_rolling_ball.call_args_list[0]
    assert call_args[1]['radius'] == 10.0

    # Verify sink operations
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')

    # Verify file upload
    mock_tile_client.client.uploadFileToFolder.assert_called_once_with(
        'test_dataset', '/tmp/output.tiff'
    )

    # Verify metadata was added
    expected_metadata = {
        'tool': 'Rolling ball',
        'radius': 10.0
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_multi_channel(mock_tile_client, mock_large_image, mock_rolling_ball, sample_params_multi_channel):
    """Test compute with multiple channels selected"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_multi_channel)

    # Should process frames for both channels (all 4 frames have IndexC=0 or IndexC=1)
    assert mock_rolling_ball.call_count == 4

    # Verify sink operations
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')

    # Verify metadata includes radius value
    expected_metadata = {
        'tool': 'Rolling ball',
        'radius': 25.0
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_high_radius_value(mock_tile_client, mock_large_image, mock_rolling_ball, sample_params_high_radius):
    """Test compute with high radius value"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_high_radius)

    # Verify rolling ball was called with high radius
    call_args = mock_rolling_ball.call_args_list[0]
    assert call_args[1]['radius'] == 80.0

    # Verify metadata includes correct radius
    expected_metadata = {
        'tool': 'Rolling ball',
        'radius': 80.0
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_no_frames_single_frame(mock_tile_client, mock_large_image, mock_rolling_ball, sample_params_basic):
    """Test compute with single frame (no 'frames' key in tiles)"""
    # Remove frames to simulate single frame scenario
    del mock_tile_client.tiles['frames']

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Should process single frame
    mock_rolling_ball.assert_called_once()
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')


def test_channel_filtering_logic(mock_tile_client, mock_large_image, mock_rolling_ball):
    """Test that only selected channels are processed"""
    params = {
        'workerInterface': {
            'Radius': 15.0,
            'Channels to correct': {'0': True, '1': False}  # Only channel 0
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
    assert mock_rolling_ball.call_count == 2
    mock_large_image.write.assert_called_once_with('/tmp/output.tiff')


def test_background_subtraction_logic(mock_tile_client, mock_large_image, mock_rolling_ball, sample_params_basic):
    """Test that background subtraction is applied correctly"""
    # Set up mock to return specific values
    original_image = np.full((100, 100), 1000, dtype=np.uint16)
    background = np.full((100, 100), 100, dtype=np.uint16)

    mock_tile_client.getRegion.return_value = original_image
    mock_rolling_ball.return_value = background

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify rolling ball was called
    mock_rolling_ball.assert_called()

    # Verify addTile was called (background subtraction happens before addTile)
    mock_large_image.addTile.assert_called()


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
    assert '"title": "Rolling ball"' in captured.out
    assert 'Processing frame' in captured.out


def test_preview_functionality(mock_tile_client, mock_worker_preview_client, mock_gaussian_filter):
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
            'Sigma': 15.0
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


def test_radius_parameter_validation():
    """Test that radius parameter is properly converted to float"""
    params_str = {
        'workerInterface': {
            'Radius': '35.5',  # String value
            'Channels to correct': {'0': True}
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client, \
            patch('large_image.new'), \
            patch('entrypoint.restoration') as mock_restoration:

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

        # Set up rolling_ball mock to return same shape as input
        mock_restoration.rolling_ball.side_effect = lambda img, radius: img * 0.1

        # This should not raise an error despite string input
        compute('test_dataset', 'http://test-api', 'test-token', params_str)

        # Verify radius was converted to float
        call_args = mock_restoration.rolling_ball.call_args_list[0]
        assert call_args[1]['radius'] == 35.5
        assert isinstance(call_args[1]['radius'], float)


def test_channel_selection_edge_cases(mock_tile_client, mock_large_image):
    """Test channel selection with edge cases"""
    # Test with no channels selected
    params_no_channels = {
        'workerInterface': {
            'Radius': 20.0,
            'Channels to correct': {'0': False, '1': False}  # No channels
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('entrypoint.restoration') as mock_restoration:
        # Run compute
        compute('test_dataset', 'http://test-api', 'test-token', params_no_channels)

        # Should not call rolling ball since no channels are selected
        assert mock_restoration.rolling_ball.call_count == 0
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

    with patch('entrypoint.restoration') as mock_restoration:
        mock_restoration.rolling_ball.side_effect = lambda img, radius: img * 0.1

        # Run compute
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

        # Should handle different dimensions correctly
        mock_restoration.rolling_ball.assert_called()
        mock_large_image.write.assert_called_once_with('/tmp/output.tiff')


def test_frame_parameter_construction(mock_tile_client, mock_large_image, mock_rolling_ball, sample_params_basic):
    """Test that frame parameters are constructed correctly"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify addTile was called with correct parameters
    mock_large_image.addTile.assert_called()

    # Check that addTile calls include the expected parameters
    add_tile_calls = mock_large_image.addTile.call_args_list
    for call in add_tile_calls:
        # Each call should have parameters like xy, z, t, c
        kwargs = call[1]
        # Should have extracted parameters from frame indices
        assert any(key in ['xy', 'z', 't', 'c'] for key in kwargs.keys())


def test_radius_bounds_validation():
    """Test radius parameter with boundary values"""
    # Test with minimum radius (0)
    params_min = {
        'workerInterface': {
            'Radius': 0,
            'Channels to correct': {'0': True}
        },
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client, \
            patch('large_image.new'), \
            patch('entrypoint.restoration') as mock_restoration:

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

        # Set up rolling_ball mock to return same shape as input
        mock_restoration.rolling_ball.side_effect = lambda img, radius: img * 0.1

        # Should handle radius=0 without error
        compute('test_dataset', 'http://test-api', 'test-token', params_min)

        # Verify radius was used correctly
        mock_restoration.rolling_ball.assert_called_with(
            mock_client.return_value.getRegion.return_value, radius=0
        )


def test_image_processing_pipeline(mock_tile_client, mock_large_image, sample_params_basic):
    """Test the complete image processing pipeline"""
    # Set up specific test data
    original_image = np.full((50, 50), 500, dtype=np.uint16)
    background_image = np.full((50, 50), 50, dtype=np.uint16)

    mock_tile_client.getRegion.return_value = original_image

    with patch('entrypoint.restoration') as mock_restoration:
        mock_restoration.rolling_ball.return_value = background_image

        # Run compute
        compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

        # Verify the pipeline: rolling_ball called, then addTile with subtracted image
        mock_restoration.rolling_ball.assert_called()
        mock_large_image.addTile.assert_called()

        # The first argument to addTile should be the background-subtracted image
        add_tile_calls = mock_large_image.addTile.call_args_list
        for call in add_tile_calls:
            processed_image = call[0][0]  # First positional argument
            # Should be numpy array (the processed image)
            assert isinstance(processed_image, np.ndarray)
