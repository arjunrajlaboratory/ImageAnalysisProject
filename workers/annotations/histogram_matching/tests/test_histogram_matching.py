import pytest
from unittest.mock import patch, MagicMock
import json
import numpy as np

# Import your worker module
from entrypoint import compute, interface


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
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 1},
            ],
            'IndexRange': {
                'IndexXY': 2,
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
        client.getRegion.return_value = np.random.randint(0, 1000, (512, 512), dtype=np.uint16)

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
def mock_match_histograms():
    """Mock the skimage.exposure.match_histograms function"""
    with patch('entrypoint.match_histograms') as mock_match:
        # Return the source image with slight modification to simulate histogram matching
        mock_match.side_effect = lambda src, ref: src * 0.9 + ref * 0.1
        yield mock_match


@pytest.fixture
def sample_params_basic():
    """Create basic sample parameters"""
    return {
        'workerInterface': {
            'Reference XY Coordinate': '1',
            'Reference Z Coordinate': '1',
            'Reference Time Coordinate': '1',
            'Channels to correct': {'0': True, '1': False}
        }
    }


@pytest.fixture
def sample_params_multi_channel():
    """Create sample parameters with multiple channels"""
    return {
        'workerInterface': {
            'Reference XY Coordinate': '2',
            'Reference Z Coordinate': '1',
            'Reference Time Coordinate': '1',
            'Channels to correct': {'0': True, '1': True}
        }
    }


@pytest.fixture
def sample_params_empty_coordinates():
    """Create sample parameters with empty coordinate strings"""
    return {
        'workerInterface': {
            'Reference XY Coordinate': '',
            'Reference Z Coordinate': '',
            'Reference Time Coordinate': '',
            'Channels to correct': {'0': True, '1': True}
        }
    }


@pytest.fixture
def sample_params_no_channels():
    """Create sample parameters with no channels selected"""
    return {
        'workerInterface': {
            'Reference XY Coordinate': '1',
            'Reference Z Coordinate': '1',
            'Reference Time Coordinate': '1',
            'Channels to correct': {'0': False, '1': False}
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
    assert 'Reference XY Coordinate' in interface_data
    assert 'Reference Z Coordinate' in interface_data
    assert 'Reference Time Coordinate' in interface_data
    assert 'Channels to correct' in interface_data

    # Check Reference XY Coordinate interface
    xy_interface = interface_data['Reference XY Coordinate']
    assert xy_interface['type'] == 'text'
    assert xy_interface['displayOrder'] == 1
    assert 'placeholder' in xy_interface['vueAttrs']
    assert xy_interface['vueAttrs']['placeholder'] == 'ex. 8'

    # Check Reference Z Coordinate interface
    z_interface = interface_data['Reference Z Coordinate']
    assert z_interface['type'] == 'text'
    assert z_interface['displayOrder'] == 2

    # Check Reference Time Coordinate interface
    time_interface = interface_data['Reference Time Coordinate']
    assert time_interface['type'] == 'text'
    assert time_interface['displayOrder'] == 3

    # Check Channels to correct interface
    channels_interface = interface_data['Channels to correct']
    assert channels_interface['type'] == 'channelCheckboxes'
    assert channels_interface['displayOrder'] == 2


def test_compute_basic_functionality(mock_tile_client, mock_large_image, mock_match_histograms, sample_params_basic):
    """Test basic compute functionality with single channel"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify coordinatesToFrameIndex was called to get reference image
    mock_tile_client.coordinatesToFrameIndex.assert_called()

    # Verify histogram matching was called for channel 0 frames (3 frames: 0, 2, 4)
    assert mock_match_histograms.call_count == 3

    # Verify sink operations
    mock_large_image.write.assert_called_once_with('/tmp/normalized.tiff')

    # Verify file upload
    mock_tile_client.client.uploadFileToFolder.assert_called_once_with(
        'test_dataset', '/tmp/normalized.tiff'
    )

    # Verify metadata was added (coordinates are 1-indexed in params, 0-indexed in metadata)
    expected_metadata = {
        'tool': 'Histogram matching',
        'reference_XY': 0,  # 1 - 1 = 0
        'reference_Z': 0,   # 1 - 1 = 0
        'reference_Time': 0  # 1 - 1 = 0
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_multi_channel(mock_tile_client, mock_large_image, mock_match_histograms, sample_params_multi_channel):
    """Test compute with multiple channels selected"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_multi_channel)

    # Should get reference images for both channels
    assert mock_tile_client.coordinatesToFrameIndex.call_count >= 2

    # Should process frames for both channels (all 6 frames have IndexC 0 or 1)
    assert mock_match_histograms.call_count == 6

    # Verify sink operations
    mock_large_image.write.assert_called_once_with('/tmp/normalized.tiff')

    # Verify metadata includes correct coordinates
    expected_metadata = {
        'tool': 'Histogram matching',
        'reference_XY': 1,  # 2 - 1 = 1
        'reference_Z': 0,   # 1 - 1 = 0
        'reference_Time': 0  # 1 - 1 = 0
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_empty_coordinates_default_to_zero(mock_tile_client, mock_large_image, mock_match_histograms, sample_params_empty_coordinates):
    """Test that empty coordinate strings default to zero"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_empty_coordinates)

    # Verify histogram matching was called for both channels (all 6 frames)
    assert mock_match_histograms.call_count == 6

    # Verify metadata uses default coordinates (0)
    expected_metadata = {
        'tool': 'Histogram matching',
        'reference_XY': 0,
        'reference_Z': 0,
        'reference_Time': 0
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_no_channels_selected_error(mock_tile_client, mock_large_image, sample_params_no_channels, capsys):
    """Test error handling when no channels are selected"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_no_channels)

    # Capture stdout to verify error was sent
    captured = capsys.readouterr()
    assert '"error": "No channels to correct"' in captured.out
    assert '"type": "error"' in captured.out

    # Should not process any frames
    mock_large_image.write.assert_not_called()


def test_compute_no_frames_error(mock_tile_client, mock_large_image, sample_params_basic, capsys):
    """Test error handling when no frames are present"""
    # Remove frames to simulate single image scenario
    del mock_tile_client.tiles['frames']

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Capture stdout to verify error was sent
    captured = capsys.readouterr()
    assert '"error": "Only one image; exiting"' in captured.out
    assert '"type": "error"' in captured.out

    # Should not write output file
    mock_large_image.write.assert_not_called()


def test_channel_filtering_logic(mock_tile_client, mock_large_image, mock_match_histograms):
    """Test that only selected channels are processed"""
    params = {
        'workerInterface': {
            'Reference XY Coordinate': '1',
            'Reference Z Coordinate': '1',
            'Reference Time Coordinate': '1',
            'Channels to correct': {'0': True, '1': False}  # Only channel 0
        }
    }

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Should only process frames with IndexC=0 (3 frames total from our mock data)
    # From our mock data: frames 0, 2, 4 have IndexC=0
    assert mock_match_histograms.call_count == 3
    mock_large_image.write.assert_called_once_with('/tmp/normalized.tiff')


def test_reference_image_retrieval(mock_tile_client, mock_large_image, mock_match_histograms, sample_params_multi_channel):
    """Test that reference images are retrieved for each channel"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_multi_channel)

    # Should call coordinatesToFrameIndex for each channel to get reference images
    # Plus additional calls during frame processing
    coordinate_calls = mock_tile_client.coordinatesToFrameIndex.call_args_list

    # Should have calls for both channels 0 and 1 to get reference images
    channel_calls = [call[0][3] for call in coordinate_calls]  # 4th argument is channel
    assert 0 in channel_calls
    assert 1 in channel_calls


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
    assert '"title": "Histogram matching"' in captured.out
    assert 'Processing frame' in captured.out


def test_coordinate_parsing_edge_cases(mock_tile_client, mock_large_image, mock_match_histograms):
    """Test coordinate parsing with various edge cases"""
    # Test with high coordinate values
    params_high_coords = {
        'workerInterface': {
            'Reference XY Coordinate': '10',
            'Reference Z Coordinate': '5',
            'Reference Time Coordinate': '3',
            'Channels to correct': {'0': True}
        }
    }

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', params_high_coords)

    # Verify metadata includes correct coordinate conversion
    expected_metadata = {
        'tool': 'Histogram matching',
        'reference_XY': 9,  # 10 - 1 = 9
        'reference_Z': 4,   # 5 - 1 = 4
        'reference_Time': 2  # 3 - 1 = 2
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_histogram_matching_application(mock_tile_client, mock_large_image, mock_match_histograms, sample_params_basic):
    """Test that histogram matching is applied correctly"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify match_histograms was called for each frame in the selected channel (3 frames)
    assert mock_match_histograms.call_count == 3

    # Check that the function was called with correct arguments
    # (source image, reference image)
    call_args = mock_match_histograms.call_args_list[0]
    assert len(call_args[0]) == 2  # Two positional arguments: source and reference


def test_reference_coordinates_bounds_handling():
    """Test that reference coordinates are handled correctly even with extreme values"""
    params_extreme = {
        'workerInterface': {
            'Reference XY Coordinate': '1',  # Will become 0 after -1
            'Reference Z Coordinate': '1',   # Will become 0 after -1
            'Reference Time Coordinate': '1',  # Will become 0 after -1
            'Channels to correct': {'0': True}
        }
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client, \
            patch('large_image.new'), \
            patch('entrypoint.match_histograms'):

        client = mock_client.return_value
        client.tiles = {
            'frames': [{'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0}],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40,
            'dtype': np.uint8
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Should not raise an error
        compute('test_dataset', 'http://test-api', 'test-token', params_extreme)

        # Verify coordinatesToFrameIndex was called with 0-indexed coordinates
        client.coordinatesToFrameIndex.assert_called_with(0, 0, 0, 0)


def test_multiple_reference_images_per_channel(mock_tile_client, mock_large_image, mock_match_histograms, sample_params_multi_channel):
    """Test that separate reference images are retrieved for each channel"""
    # Create a mock that returns different images for different channels
    def mock_get_region(*args, **kwargs):
        frame = kwargs.get('frame', 0)
        # Return different base values for different channels
        if frame % 2 == 0:  # Even frame numbers (channel 0)
            return np.full((100, 100), 100, dtype=np.uint16)
        else:  # Odd frame numbers (channel 1)
            return np.full((100, 100), 200, dtype=np.uint16)

    mock_tile_client.getRegion.side_effect = mock_get_region

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_multi_channel)

    # Should call getRegion for reference images and frame processing
    assert mock_tile_client.getRegion.call_count > 2
    mock_large_image.write.assert_called_once_with('/tmp/normalized.tiff')


def test_frame_parameter_construction(mock_tile_client, mock_large_image, mock_match_histograms, sample_params_basic):
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


def test_different_coordinate_string_formats():
    """Test handling of different coordinate string formats"""
    # Test with whitespace
    params_whitespace = {
        'workerInterface': {
            'Reference XY Coordinate': ' 2 ',
            'Reference Z Coordinate': '  1  ',
            'Reference Time Coordinate': '\t3\t',
            'Channels to correct': {'0': True}
        }
    }

    with patch('annotation_client.tiles.UPennContrastDataset') as mock_client, \
            patch('large_image.new'), \
            patch('entrypoint.match_histograms'):

        client = mock_client.return_value
        client.tiles = {
            'frames': [{'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0}],
            'channels': ['DAPI'],
            'mm_x': 1.0, 'mm_y': 1.0, 'magnification': 40,
            'dtype': np.uint8
        }
        client.getRegion.return_value = np.zeros((100, 100))
        client.coordinatesToFrameIndex.return_value = 0
        client.client = MagicMock()
        client.client.uploadFileToFolder.return_value = {'itemId': 'test'}

        # Should handle whitespace correctly
        compute('test_dataset', 'http://test-api', 'test-token', params_whitespace)

        # Verify the metadata reflects parsed coordinates
        expected_metadata = {
            'tool': 'Histogram matching',
            'reference_XY': 1,  # 2 - 1 = 1
            'reference_Z': 0,   # 1 - 1 = 0
            'reference_Time': 2  # 3 - 1 = 2
        }
        client.client.addMetadataToItem.assert_called_once_with(
            'test', expected_metadata
        )
