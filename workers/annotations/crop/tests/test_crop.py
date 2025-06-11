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
        # Set up default tile info with multiple frames
        client.tiles = {
            'frames': [
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 0, 'IndexT': 1, 'IndexC': 0},
                {'IndexXY': 0, 'IndexZ': 1, 'IndexT': 0, 'IndexC': 0},
                {'IndexXY': 1, 'IndexZ': 0, 'IndexT': 0, 'IndexC': 0},
            ],
            'IndexRange': {
                'IndexXY': 2,
                'IndexZ': 2,
                'IndexT': 2,
                'IndexC': 1
            },
            'channels': ['DAPI'],
            'mm_x': 0.65,
            'mm_y': 0.65,
            'magnification': 20
        }

        # Mock getRegion to return dummy image data
        client.getRegion.return_value = np.random.randint(0, 255, (512, 512, 1), dtype=np.uint8)

        # Mock the girder client
        mock_gc = MagicMock()
        mock_gc.uploadFileToFolder.return_value = {'itemId': 'test_item_id'}
        client.client = mock_gc

        yield client


@pytest.fixture
def mock_annotation_client():
    """Mock the annotations.UPennContrastAnnotationClient"""
    with patch('annotation_client.annotations.UPennContrastAnnotationClient') as mock_client:
        client = mock_client.return_value

        # Mock annotations with crop rectangle
        client.getAnnotationsByDatasetId.return_value = [
            {
                'id': 'test_rect_id',
                'coordinates': [
                    {'x': 100, 'y': 100},
                    {'x': 200, 'y': 100},
                    {'x': 200, 'y': 200},
                    {'x': 100, 'y': 200}
                ],
                'tags': ['crop_region']
            }
        ]

        yield client


@pytest.fixture
def mock_worker_preview_client():
    """Mock the UPennContrastWorkerPreviewClient"""
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_client:
        yield mock_client.return_value


@pytest.fixture
def mock_annotation_tools():
    """Mock the annotation_tools.get_annotations_with_tags"""
    with patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_tools:
        mock_tools.return_value = [
            {
                'id': 'test_rect_id',
                'coordinates': [
                    {'x': 100, 'y': 100},
                    {'x': 200, 'y': 100},
                    {'x': 200, 'y': 200},
                    {'x': 100, 'y': 200}
                ],
                'tags': ['crop_region']
            }
        ]
        yield mock_tools


@pytest.fixture
def mock_batch_argument_parser():
    """Mock the batch_argument_parser.process_range_list"""
    with patch('annotation_utilities.batch_argument_parser.process_range_list') as mock_parser:
        # Default behavior - return what was passed in
        mock_parser.side_effect = lambda x, convert_one_to_zero_index=False: [
            int(i) for i in x.split(',')]
        yield mock_parser


@pytest.fixture
def mock_large_image():
    """Mock large_image operations"""
    with patch('large_image.new') as mock_li_new:
        mock_sink = MagicMock()
        mock_li_new.return_value = mock_sink
        yield mock_sink


@pytest.fixture
def sample_params_basic():
    """Create basic sample parameters"""
    return {
        'workerInterface': {
            'XY Range': '',
            'Z Range': '',
            'Time Range': '',
            'Crop Rectangle': None
        }
    }


@pytest.fixture
def sample_params_with_ranges():
    """Create sample parameters with specific ranges"""
    return {
        'workerInterface': {
            'XY Range': '0,1',
            'Z Range': '0',
            'Time Range': '0,1',
            'Crop Rectangle': None
        }
    }


@pytest.fixture
def sample_params_with_crop():
    """Create sample parameters with crop rectangle"""
    return {
        'workerInterface': {
            'XY Range': '',
            'Z Range': '',
            'Time Range': '',
            'Crop Rectangle': ['crop_region']
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
    assert 'XY Range' in interface_data
    assert 'Z Range' in interface_data
    assert 'Time Range' in interface_data
    assert 'Crop Rectangle' in interface_data

    # Check XY Range interface
    xy_range = interface_data['XY Range']
    assert xy_range['type'] == 'text'
    assert xy_range['displayOrder'] == 1
    assert 'placeholder' in xy_range['vueAttrs']
    assert xy_range['vueAttrs']['placeholder'] == 'ex. 1-3, 5-8'

    # Check Z Range interface
    z_range = interface_data['Z Range']
    assert z_range['type'] == 'text'
    assert z_range['displayOrder'] == 2

    # Check Time Range interface
    time_range = interface_data['Time Range']
    assert time_range['type'] == 'text'
    assert time_range['displayOrder'] == 3

    # Check Crop Rectangle interface
    crop_rect = interface_data['Crop Rectangle']
    assert crop_rect['type'] == 'tags'
    assert crop_rect['displayOrder'] == 4


def test_compute_basic_no_ranges(mock_tile_client, mock_annotation_client, mock_large_image, sample_params_basic):
    """Test basic compute functionality with no specific ranges"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify that all frames were processed (4 frames in mock data)
    assert mock_tile_client.getRegion.call_count == 4

    # Verify sink operations
    mock_large_image.write.assert_called_once_with('/tmp/cropped.tiff')

    # Verify file upload
    mock_tile_client.client.uploadFileToFolder.assert_called_once_with(
        'test_dataset', '/tmp/cropped.tiff'
    )

    # Verify metadata was added
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', {'tool': 'Crop'}
    )


def test_compute_with_specific_ranges(mock_tile_client, mock_annotation_client, mock_large_image,
                                      mock_batch_argument_parser, sample_params_with_ranges):
    """Test compute with specific XY, Z, and Time ranges"""
    # Set up the parser to return specific ranges
    mock_batch_argument_parser.side_effect = lambda x, convert_one_to_zero_index=False: [
        0, 1] if '0,1' in x else [0]

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_with_ranges)

    # Should process only matching frames
    assert mock_tile_client.getRegion.call_count > 0
    mock_large_image.write.assert_called_once_with('/tmp/cropped.tiff')


def test_compute_with_crop_rectangle(mock_tile_client, mock_annotation_client, mock_large_image,
                                     mock_annotation_tools, sample_params_with_crop):
    """Test compute with crop rectangle specified"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_with_crop)

    # Verify annotations were queried
    mock_annotation_client.getAnnotationsByDatasetId.assert_called()

    # Verify annotation tools were used to filter by tags
    mock_annotation_tools.assert_called_once()

    # Verify getRegion was called with crop parameters
    mock_tile_client.getRegion.assert_called()

    # Check that crop parameters were used (left=100, top=100, right=200, bottom=200)
    call_args = mock_tile_client.getRegion.call_args_list[0]
    assert 'left' in call_args[1]
    assert 'top' in call_args[1]
    assert 'right' in call_args[1]
    assert 'bottom' in call_args[1]
    assert call_args[1]['left'] == 100
    assert call_args[1]['top'] == 100
    assert call_args[1]['right'] == 200
    assert call_args[1]['bottom'] == 200

    # Verify metadata includes crop dimensions
    expected_metadata = {
        'tool': 'Crop',
        'crop left': 100,
        'crop top': 100,
        'crop right': 200,
        'crop bottom': 200
    }
    mock_tile_client.client.addMetadataToItem.assert_called_once_with(
        'test_item_id', expected_metadata
    )


def test_compute_no_crop_rectangle_found(mock_tile_client, mock_annotation_client, mock_annotation_tools,
                                         sample_params_with_crop, capsys):
    """Test error handling when crop rectangle is not found"""
    # Mock annotation tools to return empty list
    mock_annotation_tools.return_value = []

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_with_crop)

    # Capture stdout to verify error was sent
    captured = capsys.readouterr()
    assert '"error": "No crop rectangle found"' in captured.out
    assert '"type": "error"' in captured.out


def test_compute_no_frames_single_frame(mock_tile_client, mock_annotation_client, mock_large_image, sample_params_basic):
    """Test compute with single frame (no 'frames' key in tiles)"""
    # Remove frames to simulate single frame scenario
    del mock_tile_client.tiles['frames']

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Should process single frame
    mock_tile_client.getRegion.assert_called_once_with('test_dataset', frame=0)
    mock_large_image.write.assert_called_once_with('/tmp/cropped.tiff')


def test_compute_no_index_range(mock_tile_client, mock_annotation_client, mock_large_image, sample_params_basic):
    """Test compute with no IndexRange in tiles"""
    # Remove IndexRange to test fallback behavior
    del mock_tile_client.tiles['IndexRange']

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Should still process frames with default ranges
    mock_large_image.write.assert_called_once_with('/tmp/cropped.tiff')


def test_range_parsing_with_gaps(mock_tile_client, mock_annotation_client, mock_large_image,
                                 mock_batch_argument_parser):
    """Test range parsing with gaps (e.g., '1-3,5-8')"""
    params = {
        'workerInterface': {
            'XY Range': '1-3,5-8',
            'Z Range': '0,2',
            'Time Range': '1,3',
            'Crop Rectangle': None
        }
    }

    # Mock parser to return specific ranges
    mock_batch_argument_parser.side_effect = lambda x, convert_one_to_zero_index=False: {
        '1-3,5-8': [0, 1, 2, 4, 5, 6, 7],  # Convert to 0-indexed
        '0,2': [0, 1],
        '1,3': [0, 2]
    }.get(x, [0])

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Verify parser was called for each range
    assert mock_batch_argument_parser.call_count == 3
    mock_large_image.write.assert_called_once_with('/tmp/cropped.tiff')


def test_channel_metadata_preservation(mock_tile_client, mock_annotation_client, mock_large_image, sample_params_basic):
    """Test that channel metadata is preserved in output"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Verify channel names were set on sink
    assert mock_large_image.channelNames == ['DAPI']

    # Verify other metadata was preserved
    assert mock_large_image.mm_x == 0.65
    assert mock_large_image.mm_y == 0.65
    assert mock_large_image.magnification == 20


def test_complex_crop_rectangle_coordinates(mock_tile_client, mock_annotation_client, mock_large_image, mock_annotation_tools):
    """Test crop rectangle with complex polygon coordinates"""
    # Mock annotation with irregular polygon
    mock_annotation_tools.return_value = [
        {
            'id': 'test_polygon_id',
            'coordinates': [
                {'x': 50, 'y': 75},
                {'x': 150, 'y': 50},
                {'x': 200, 'y': 100},
                {'x': 175, 'y': 200},
                {'x': 75, 'y': 175}
            ],
            'tags': ['crop_region']
        }
    ]

    params = {
        'workerInterface': {
            'XY Range': '',
            'Z Range': '',
            'Time Range': '',
            'Crop Rectangle': ['crop_region']
        }
    }

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Verify bounding box calculation (min/max of coordinates)
    call_args = mock_tile_client.getRegion.call_args_list[0]
    assert call_args[1]['left'] == 50    # min x
    assert call_args[1]['top'] == 50     # min y
    assert call_args[1]['right'] == 200  # max x
    assert call_args[1]['bottom'] == 200  # max y


def test_empty_range_strings(mock_tile_client, mock_annotation_client, mock_large_image):
    """Test handling of empty or whitespace-only range strings"""
    params = {
        'workerInterface': {
            'XY Range': '   ',  # whitespace only
            'Z Range': '',      # empty
            'Time Range': None,  # None
            'Crop Rectangle': None
        }
    }

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Should use default ranges and complete successfully
    mock_large_image.write.assert_called_once_with('/tmp/cropped.tiff')


def test_progress_reporting(mock_tile_client, mock_annotation_client, mock_large_image, sample_params_basic, capsys):
    """Test that progress is reported during processing"""
    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_basic)

    # Capture stdout to verify progress was sent
    captured = capsys.readouterr()
    assert '"progress":' in captured.out
    assert '"title": "Crop"' in captured.out
    assert 'Processing frame' in captured.out


def test_frame_filtering_logic(mock_tile_client, mock_annotation_client, mock_large_image, mock_batch_argument_parser):
    """Test that frame filtering logic works correctly"""
    # Set up specific ranges that should filter out some frames
    params = {
        'workerInterface': {
            'XY Range': '0',     # Only XY=0
            'Z Range': '0',      # Only Z=0
            'Time Range': '0',   # Only T=0
            'Crop Rectangle': None
        }
    }

    mock_batch_argument_parser.side_effect = lambda x, convert_one_to_zero_index=False: [0]

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Should only process frames matching XY=0, Z=0, T=0
    # From our mock data, that's only the first frame
    assert mock_tile_client.getRegion.call_count == 1
    mock_large_image.write.assert_called_once_with('/tmp/cropped.tiff')


def test_mixed_annotation_types(mock_tile_client, mock_annotation_client, mock_large_image, mock_annotation_tools):
    """Test handling of mixed polygon and rectangle annotations"""
    # Mock both polygon and rectangle annotations
    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        [{'id': 'poly1', 'coordinates': [{'x': 10, 'y': 10}], 'tags': ['crop_region']}],  # polygons
        [{'id': 'rect1', 'coordinates': [{'x': 20, 'y': 20}], 'tags': ['crop_region']}]   # rectangles
    ]

    mock_annotation_tools.return_value = [
        {
            'id': 'combined_annotation',
            'coordinates': [
                {'x': 10, 'y': 10},
                {'x': 20, 'y': 20}
            ],
            'tags': ['crop_region']
        }
    ]

    params = {
        'workerInterface': {
            'XY Range': '',
            'Z Range': '',
            'Time Range': '',
            'Crop Rectangle': ['crop_region']
        }
    }

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', params)

    # Verify both polygon and rectangle annotations were queried
    assert mock_annotation_client.getAnnotationsByDatasetId.call_count == 2

    # Check calls were made for both shapes
    call_args_list = mock_annotation_client.getAnnotationsByDatasetId.call_args_list
    shapes_requested = [call[1]['shape'] for call in call_args_list]
    assert 'polygon' in shapes_requested
    assert 'rectangle' in shapes_requested

    mock_large_image.write.assert_called_once_with('/tmp/cropped.tiff')
