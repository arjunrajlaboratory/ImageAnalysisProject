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
        'image': 'properties/blob_intensity_percentile:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Channel': 0,
            'Percentile': 75  # Add percentile parameter
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
        assert 'Percentile' in interface_data
        assert interface_data['Percentile']['type'] == 'number'
        assert interface_data['Percentile']['default'] == 50


def test_worker_startup(mock_worker_client, mock_dataset_client, sample_params):
    """Test that the worker starts up correctly with no annotations"""
    # Run computation with empty annotation list
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify that get_annotation_list_by_shape was called
    mock_worker_client.get_annotation_list_by_shape.assert_called_once_with(
        'polygon', limit=0)

    # Since there are no annotations, add_multiple_annotation_property_values should not be called
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_uniform_intensity_percentile(mock_worker_client, mock_dataset_client, sample_params):
    """Test percentile calculation with a uniform intensity image"""
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

    # For a uniform image with all pixels = 50, the percentile value should be 50
    percentile = float(sample_params['workerInterface']['Percentile'])
    prop_name = f'{percentile}thPercentileIntensity'

    # Assert that the property exists and has the expected value
    assert prop_name in property_values
    assert property_values[prop_name] == pytest.approx(50.0)


def test_gradient_intensity_percentile(mock_worker_client, mock_dataset_client, sample_params):
    """Test percentile calculation with a gradient intensity image"""
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

    # Extract the exact region from our test image to calculate expected values
    region = test_image[5:15, 5:15]
    percentile = float(sample_params['workerInterface']['Percentile'])
    expected_percentile = np.percentile(region, percentile)
    prop_name = f'{percentile}thPercentileIntensity'

    # Verify the computed metrics match our expectations
    assert prop_name in property_values
    assert property_values[prop_name] == pytest.approx(expected_percentile)


def test_multiple_annotations_percentile(mock_worker_client, mock_dataset_client, sample_params):
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
    percentile = float(sample_params['workerInterface']['Percentile'])
    prop_name = f'{percentile}thPercentileIntensity'

    # Verify square metrics
    square_values = property_values['test_square']
    assert prop_name in square_values
    assert square_values[prop_name] == pytest.approx(np.percentile(square_region, percentile))

    # Verify rectangle metrics
    rectangle_values = property_values['test_rectangle']
    assert prop_name in rectangle_values
    assert rectangle_values[prop_name] == pytest.approx(np.percentile(rectangle_region, percentile))


def test_different_percentiles(mock_worker_client, mock_dataset_client, sample_params):
    """Test different percentile values"""
    # Create a test image with a range of values
    test_image = np.arange(0, 100, dtype=np.uint8).reshape(10, 10)

    # Create a test annotation covering the whole image
    test_annotation = {
        '_id': 'test_percentile',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0},
            {'x': 0, 'y': 0}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }

    # Set up mock to return our test annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [test_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Test different percentile values
    percentiles_to_test = [0.0, 25.0, 50.0, 75.0, 90.0, 99.0]

    for percentile in percentiles_to_test:
        # Update the percentile parameter
        sample_params['workerInterface']['Percentile'] = percentile

        # Run computation
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Get the property values that were sent to the server
        calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
        assert len(calls) > 0

        # Get the computed metrics
        property_values = calls[-1][0][0]['test_dataset']['test_percentile']

        # Calculate expected percentile
        expected_percentile = np.percentile(test_image, percentile)
        prop_name = f'{percentile}thPercentileIntensity'

        # Verify the computed metrics match our expectations
        assert prop_name in property_values
        assert property_values[prop_name] == pytest.approx(expected_percentile)


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
    percentile = float(sample_params['workerInterface']['Percentile'])
    prop_name = f'{percentile}thPercentileIntensity'
    valid_values = property_values['valid_annotation']
    assert prop_name in valid_values
    assert valid_values[prop_name] == pytest.approx(100.0)
