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
