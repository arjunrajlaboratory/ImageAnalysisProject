import pytest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the worker module
from entrypoint import compute, interface


@pytest.fixture
def mock_worker_client():
    """Mock the UPennContrastWorkerClient"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerClient'
    ) as mock_client:
        client = mock_client.return_value
        # Set up default behaviors
        client.get_annotation_list_by_shape.return_value = []
        yield client


@pytest.fixture
def mock_dataset_client():
    """Mock the UPennContrastDataset"""
    with patch(
        'annotation_client.tiles.UPennContrastDataset'
    ) as mock_client:
        client = mock_client.return_value
        # Set up default behaviors
        client.getRegion.return_value = None
        yield client


@pytest.fixture
def sample_annotation():
    """Create a sample polygon annotation"""
    return {
        '_id': 'test_id_1',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0},
            {'x': 0, 'y': 0}  # Close the polygon
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['cell']  # Add a tag that matches our filter
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'test_intensity',
        'image': 'properties/blob_intensity:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Channel': 0
        }
    }


def test_interface():
    """Test the interface generation"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        # Verify interface was set
        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        # Verify interface structure
        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]
        assert 'Channel' in interface_data
        assert interface_data['Channel']['type'] == 'channel'
        assert interface_data['Channel']['required'] is True


def test_worker_startup(mock_worker_client, mock_dataset_client, sample_params):
    """Test that the worker starts up correctly with no annotations"""
    # Run computation with empty annotation list
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify that get_annotation_list_by_shape was called
    mock_worker_client.get_annotation_list_by_shape.assert_called_once_with(
        'polygon', limit=0)

    # Since there are no annotations, add_multiple_annotation_property_values should not be called
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_uniform_intensity_calculation(mock_worker_client, mock_dataset_client, sample_params):
    """Test intensity calculations with a uniform intensity image"""
    # Create a uniform intensity test image (all pixels = 50)
    test_image = np.ones((20, 20), dtype=np.uint8) * 50

    # Create a test annotation (10x10 square)
    test_annotation = {
        '_id': 'test_square',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 5},
            {'x': 5, 'y': 5}  # Close the polygon
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['cell']
    }

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [test_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']['test_square']

    # For a uniform image with all pixels = 50, all metrics should be 50
    # except total intensity which is 50 * number of pixels
    assert property_values['MeanIntensity'] == pytest.approx(50.0)
    assert property_values['MaxIntensity'] == pytest.approx(50.0)
    assert property_values['MinIntensity'] == pytest.approx(50.0)
    assert property_values['MedianIntensity'] == pytest.approx(50.0)
    assert property_values['25thPercentileIntensity'] == pytest.approx(50.0)
    assert property_values['75thPercentileIntensity'] == pytest.approx(50.0)

    # The square is 10x10 = 100 pixels, so total intensity should be 50 * 100 = 5000
    assert property_values['TotalIntensity'] == pytest.approx(5000.0)


def test_gradient_intensity_calculation(mock_worker_client, mock_dataset_client, sample_params):
    """Test intensity calculations with a gradient intensity image"""
    # Create a gradient test image (values from 0 to 99)
    y, x = np.mgrid[0:20, 0:20]
    test_image = (x + y) * 2.5  # Creates a diagonal gradient from 0 to 95
    test_image = test_image.astype(np.uint8)

    # Create a test annotation (10x10 square in the middle)
    test_annotation = {
        '_id': 'test_gradient_square',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 5},
            {'x': 5, 'y': 5}  # Close the polygon
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['cell']
    }

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [test_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']['test_gradient_square']

    # Calculate expected values for the 10x10 square from (5,5) to (15,15)
    # For this region in our gradient:
    # Min value should be at (5,5) = (5+5)*2.5 = 25
    # Max value should be at (14,14) = (14+14)*2.5 = 70
    # Mean should be the average of all values in the square

    # Extract the exact region from our test image to calculate expected values
    region = test_image[5:15, 5:15]
    expected_mean = np.mean(region)
    expected_max = np.max(region)
    expected_min = np.min(region)
    expected_median = np.median(region)
    expected_q25 = np.percentile(region, 25)
    expected_q75 = np.percentile(region, 75)
    expected_total = np.sum(region)

    # Verify the computed metrics match our expectations
    assert property_values['MeanIntensity'] == pytest.approx(expected_mean)
    assert property_values['MaxIntensity'] == pytest.approx(expected_max)
    assert property_values['MinIntensity'] == pytest.approx(expected_min)
    assert property_values['MedianIntensity'] == pytest.approx(expected_median)
    assert property_values['25thPercentileIntensity'] == pytest.approx(expected_q25)
    assert property_values['75thPercentileIntensity'] == pytest.approx(expected_q75)
    assert property_values['TotalIntensity'] == pytest.approx(expected_total)


def test_multiple_annotations(mock_worker_client, mock_dataset_client, sample_params):
    """Test processing multiple annotations with different shapes"""
    # Create a test image with a gradient
    test_image = np.zeros((30, 30), dtype=np.uint8)
    for i in range(30):
        for j in range(30):
            test_image[i, j] = (i + j) * 2  # Simple gradient

    # Create multiple test annotations with different shapes
    square_annotation = {
        '_id': 'test_square',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 5},
            {'x': 5, 'y': 5}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    rectangle_annotation = {
        '_id': 'test_rectangle',
        'coordinates': [
            {'x': 20, 'y': 5},
            {'x': 20, 'y': 25},
            {'x': 25, 'y': 25},
            {'x': 25, 'y': 5},
            {'x': 20, 'y': 5}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Set up mock to return our test annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        square_annotation, rectangle_annotation
    ]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics for both annotations
    property_values = calls[0][0][0]['test_dataset']
    assert len(property_values) == 2
    assert 'test_square' in property_values
    assert 'test_rectangle' in property_values

    # Extract the regions from our test image to calculate expected values
    square_region = test_image[5:15, 5:15]
    rectangle_region = test_image[5:25, 20:25]

    # Verify square metrics
    square_values = property_values['test_square']
    assert square_values['MeanIntensity'] == pytest.approx(np.mean(square_region))
    assert square_values['MaxIntensity'] == pytest.approx(np.max(square_region))
    assert square_values['MinIntensity'] == pytest.approx(np.min(square_region))

    # Verify rectangle metrics
    rectangle_values = property_values['test_rectangle']
    assert rectangle_values['MeanIntensity'] == pytest.approx(np.mean(rectangle_region))
    assert rectangle_values['MaxIntensity'] == pytest.approx(np.max(rectangle_region))
    assert rectangle_values['MinIntensity'] == pytest.approx(np.min(rectangle_region))


def test_edge_cases(mock_worker_client, mock_dataset_client, sample_params):
    """Test edge cases like empty regions or annotations outside image bounds"""
    # Create a test image
    test_image = np.ones((20, 20), dtype=np.uint8) * 100

    # Create an annotation with no pixels inside (too small)
    empty_annotation = {
        '_id': 'empty_annotation',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 5.1},
            {'x': 5.1, 'y': 5.1},
            {'x': 5.1, 'y': 5},
            {'x': 5, 'y': 5}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Create an annotation outside image bounds
    outside_annotation = {
        '_id': 'outside_annotation',
        'coordinates': [
            {'x': 25, 'y': 25},
            {'x': 25, 'y': 35},
            {'x': 35, 'y': 35},
            {'x': 35, 'y': 25},
            {'x': 25, 'y': 25}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Create a valid annotation for comparison
    valid_annotation = {
        '_id': 'valid_annotation',
        'coordinates': [
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 10},
            {'x': 10, 'y': 10}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Set up mock to return our test annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        empty_annotation, outside_annotation, valid_annotation
    ]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']

    # Only the valid annotation should have metrics
    assert 'valid_annotation' in property_values
    assert 'empty_annotation' not in property_values
    assert 'outside_annotation' not in property_values

    # Verify valid annotation metrics
    valid_values = property_values['valid_annotation']
    assert valid_values['MeanIntensity'] == pytest.approx(100.0)
    assert valid_values['MaxIntensity'] == pytest.approx(100.0)
    assert valid_values['MinIntensity'] == pytest.approx(100.0)


def test_less_than_3_coordinates_warning(mock_worker_client, mock_dataset_client, sample_params, capsys):
    """Test that a warning is issued when an annotation has less than 3 coordinates"""
    # Create a test image
    test_image = np.ones((20, 20), dtype=np.uint8) * 100

    # Create an annotation with only 2 coordinates (invalid polygon)
    invalid_annotation = {
        '_id': 'invalid_annotation',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 10},
            {'x': 5, 'y': 5}  # Closing point doesn't count as a new vertex
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [invalid_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Capture the stdout output
    captured = capsys.readouterr()

    # Check if the warning message about no pixels in mask is in the output
    assert '"warning": "No pixels in mask"' in captured.out
    assert '"info": "Object invalid_annotation has no pixels in the mask."' in captured.out

    # Verify that add_multiple_annotation_property_values was called with an empty dictionary
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1
    assert calls[0][0][0] == {'test_dataset': {}}


def test_z_planes_intensity_calculation(mock_worker_client, mock_dataset_client, sample_params):
    """Test intensity calculations across multiple Z planes"""
    # Create two test images with different intensities for different Z planes
    test_image_z0 = np.ones((20, 20), dtype=np.uint8) * 50  # Z=0 (plane 1) has intensity 50
    test_image_z1 = np.ones((20, 20), dtype=np.uint8) * 100  # Z=1 (plane 2) has intensity 100

    # Create a test annotation (10x10 square)
    test_annotation = {
        '_id': 'test_z_planes',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 5},
            {'x': 5, 'y': 5}  # Close the polygon
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['cell']
    }

    # Set up the parameters with Z planes specified
    z_params = sample_params.copy()
    z_params['workerInterface'] = z_params.get('workerInterface', {}).copy()
    z_params['workerInterface']['Z planes'] = '1-2'  # Specifying Z planes 1-2 (0-indexed: 0-1)

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [test_annotation]

    # Set up mock tile info for IndexRange
    mock_dataset_client.tiles = {
        'IndexRange': {
            'IndexZ': 2  # We have 2 Z planes (0 and 1)
        }
    }

    # Set up mock to return different images based on frame coordinates
    def get_region_side_effect(dataset_id, frame):
        if frame == 0:  # First frame (Z=0)
            return test_image_z0
        elif frame == 1:  # Second frame (Z=1)
            return test_image_z1
        return None

    mock_dataset_client.getRegion.side_effect = get_region_side_effect

    # Set up mock to convert coordinates to frame index
    def coordinates_to_frame_side_effect(xy, z, time, channel):
        # Return frame index based on Z coordinate
        return z

    mock_dataset_client.coordinatesToFrameIndex.side_effect = coordinates_to_frame_side_effect

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', z_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']['test_z_planes']

    # Verify that we have a nested dictionary with z001 and z002 keys
    assert 'MeanIntensity' in property_values
    assert 'z001' in property_values['MeanIntensity']
    assert 'z002' in property_values['MeanIntensity']

    # Verify that the intensity values are correct for each plane
    assert property_values['MeanIntensity']['z001'] == pytest.approx(50.0)
    assert property_values['MeanIntensity']['z002'] == pytest.approx(100.0)

    assert property_values['MaxIntensity']['z001'] == pytest.approx(50.0)
    assert property_values['MaxIntensity']['z002'] == pytest.approx(100.0)

    assert property_values['MinIntensity']['z001'] == pytest.approx(50.0)
    assert property_values['MinIntensity']['z002'] == pytest.approx(100.0)

    assert property_values['MedianIntensity']['z001'] == pytest.approx(50.0)
    assert property_values['MedianIntensity']['z002'] == pytest.approx(100.0)

    assert property_values['25thPercentileIntensity']['z001'] == pytest.approx(50.0)
    assert property_values['25thPercentileIntensity']['z002'] == pytest.approx(100.0)

    assert property_values['75thPercentileIntensity']['z001'] == pytest.approx(50.0)
    assert property_values['75thPercentileIntensity']['z002'] == pytest.approx(100.0)

    # For a 10x10 square, total intensity should be 50 * 100 = 5000 for z001 and 100 * 100 = 10000 for z002
    assert property_values['TotalIntensity']['z001'] == pytest.approx(5000.0)
    assert property_values['TotalIntensity']['z002'] == pytest.approx(10000.0)
