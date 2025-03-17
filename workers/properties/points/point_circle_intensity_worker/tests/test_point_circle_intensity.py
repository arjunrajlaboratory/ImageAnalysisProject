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
        client.coordinatesToFrameIndex.return_value = 0
        yield client


@pytest.fixture
def mock_annotation_tools():
    """Mock the annotation_tools module"""
    with patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_annotations:
        # Set up default behaviors
        mock_get_annotations.return_value = []
        yield mock_get_annotations


@pytest.fixture
def sample_point_annotation():
    """Create a sample point annotation"""
    return {
        '_id': 'test_point_1',
        'coordinates': [
            {'x': 10, 'y': 10}
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['nucleus']  # Add the tag that matches our filter
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'Point Circle Intensity',
        'image': 'properties/point_circle_intensity:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'point',
        'workerInterface': {
            'Channel': 0,
            'Radius': 3
        }
    }


def test_interface():
    """Test the interface generation for point circle intensity"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        # Verify interface was set
        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        # Verify interface structure
        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]
        assert 'Point Intensity' in interface_data
        assert interface_data['Point Intensity']['type'] == 'notes'

        assert 'Channel' in interface_data
        assert interface_data['Channel']['type'] == 'channel'
        assert interface_data['Channel']['required'] is True

        assert 'Radius' in interface_data
        assert interface_data['Radius']['type'] == 'number'
        assert interface_data['Radius']['min'] == 0.5
        assert interface_data['Radius']['max'] == 10
        assert interface_data['Radius']['default'] == 1


def test_worker_startup(mock_worker_client, mock_dataset_client, mock_annotation_tools, sample_params):
    """Test that the worker starts up correctly with no annotations"""
    # Run computation with empty annotation list
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify that get_annotation_list_by_shape was called with 'point'
    mock_worker_client.get_annotation_list_by_shape.assert_called_once_with(
        'point', limit=0)

    # Verify that get_annotations_with_tags was called
    mock_annotation_tools.assert_called_once()

    # Since there are no annotations, add_multiple_annotation_property_values should not be called
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_uniform_intensity_calculation(mock_worker_client, mock_dataset_client, mock_annotation_tools, sample_params, sample_point_annotation):
    """Test intensity calculations with a uniform intensity image"""
    # Create a uniform intensity test image (all pixels = 50)
    test_image = np.ones((20, 20), dtype=np.uint8) * 50

    # Set up mock to return our point annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_point_annotation]
    mock_annotation_tools.return_value = [sample_point_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']['test_point_1']

    # For a uniform image with all pixels = 50, all metrics should be 50
    # except total intensity which depends on the number of pixels in the circle
    assert property_values['MeanIntensity'] == pytest.approx(50.0)
    assert property_values['MaxIntensity'] == pytest.approx(50.0)
    assert property_values['MinIntensity'] == pytest.approx(50.0)
    assert property_values['MedianIntensity'] == pytest.approx(50.0)
    assert property_values['25thPercentileIntensity'] == pytest.approx(50.0)
    assert property_values['75thPercentileIntensity'] == pytest.approx(50.0)

    # The total intensity should be 50 * number of pixels in the circle
    # For a radius of 3, the circle should have approximately π*r² = π*3² ≈ 28 pixels
    # So total intensity should be approximately 50 * 28 = 1400
    assert property_values['TotalIntensity'] > 0  # Just check it's positive for now


def test_gradient_intensity_calculation(mock_worker_client, mock_dataset_client, mock_annotation_tools, sample_params):
    """Test intensity calculations with a gradient intensity image"""
    # Create a gradient test image (values from 0 to 99)
    y, x = np.mgrid[0:20, 0:20]
    test_image = (x + y) * 2.5  # Creates a diagonal gradient from 0 to 95
    test_image = test_image.astype(np.uint8)

    # Create a point annotation in the middle of the gradient
    point_annotation = {
        '_id': 'test_gradient_point',
        'coordinates': [
            {'x': 10, 'y': 10}
        ],
        'location': {
            'Time': 0,
            'Z': 0,
            'XY': 0
        },
        'tags': ['nucleus']
    }

    # Set up mock to return our point annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [point_annotation]
    mock_annotation_tools.return_value = [point_annotation]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']['test_gradient_point']

    # Calculate expected values for the circle around point (10, 10)
    # The value at (10, 10) in our gradient is (10+10)*2.5 = 50
    # For a radius of 3, we'll have a range of values in the circle

    # Extract the region around our point to calculate expected values
    # We can't easily extract the exact circle, so we'll just verify the metrics are reasonable
    assert property_values['MeanIntensity'] > 0
    assert property_values['MaxIntensity'] > property_values['MeanIntensity']
    assert property_values['MinIntensity'] < property_values['MeanIntensity']
    assert property_values['MedianIntensity'] > 0
    assert property_values['25thPercentileIntensity'] < property_values['MedianIntensity']
    assert property_values['75thPercentileIntensity'] > property_values['MedianIntensity']
    assert property_values['TotalIntensity'] > 0


def test_multiple_points(mock_worker_client, mock_dataset_client, mock_annotation_tools, sample_params):
    """Test processing multiple point annotations"""
    # Create a test image with a gradient
    test_image = np.zeros((30, 30), dtype=np.uint8)
    for i in range(30):
        for j in range(30):
            test_image[i, j] = (i + j) * 2  # Simple gradient

    # Create multiple point annotations at different locations
    point1 = {
        '_id': 'test_point_1',
        'coordinates': [{'x': 5, 'y': 5}],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['nucleus']
    }

    point2 = {
        '_id': 'test_point_2',
        'coordinates': [{'x': 15, 'y': 15}],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['nucleus']
    }

    point3 = {
        '_id': 'test_point_3',
        'coordinates': [{'x': 25, 'y': 25}],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['nucleus']
    }

    # Set up mock to return our test annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [point1, point2, point3]
    mock_annotation_tools.return_value = [point1, point2, point3]

    # Set up mock to return our test image
    mock_dataset_client.getRegion.return_value = test_image
    mock_dataset_client.coordinatesToFrameIndex.return_value = 0

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics for all points
    property_values = calls[0][0][0]['test_dataset']
    assert len(property_values) == 3
    assert 'test_point_1' in property_values
    assert 'test_point_2' in property_values
    assert 'test_point_3' in property_values

    # Verify that the intensity values increase as we move along the gradient
    assert property_values['test_point_1']['MeanIntensity'] < property_values['test_point_2']['MeanIntensity']
    assert property_values['test_point_2']['MeanIntensity'] < property_values['test_point_3']['MeanIntensity']


def test_edge_cases(mock_worker_client, mock_dataset_client, mock_annotation_tools, sample_params):
    """Test edge cases like points outside image bounds"""
    # Create a test image
    test_image = np.ones((20, 20), dtype=np.uint8) * 100

    # Create a point outside image bounds
    outside_point = {
        '_id': 'outside_point',
        'coordinates': [{'x': 25, 'y': 25}],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['nucleus']
    }

    # Create a point at the edge of the image
    edge_point = {
        '_id': 'edge_point',
        'coordinates': [{'x': 19, 'y': 19}],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['nucleus']
    }

    # Create a valid point in the middle of the image
    valid_point = {
        '_id': 'valid_point',
        'coordinates': [{'x': 10, 'y': 10}],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['nucleus']
    }

    # Set up mock to return our test annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        outside_point, edge_point, valid_point
    ]
    mock_annotation_tools.return_value = [outside_point, edge_point, valid_point]

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

    # The valid point should have metrics
    assert 'valid_point' in property_values
    assert property_values['valid_point']['MeanIntensity'] == pytest.approx(100.0)

    # The edge point should have metrics, but some pixels in its circle will be outside the image
    assert 'edge_point' in property_values

    # The outside point might not have metrics if the circle doesn't intersect the image
    # This depends on the implementation, so we won't assert anything about it
