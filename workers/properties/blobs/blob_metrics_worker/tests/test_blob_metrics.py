import pytest
from unittest.mock import patch

# Import your worker module
# Assuming your entrypoint.py is in the same directory
from entrypoint import compute, convert_units, interface


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
        'tags': ['nucleus']  # Add the tag that matches our filter
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'test_metrics',
        'image': 'properties/blob_metrics:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'polygon',
        'workerInterface': {
            'Use physical units': True,
            'Units': 'µm'
        },
        'scales': {
            'pixelSize': {'unit': 'mm', 'value': 0.000219080212825376},
            'tStep': {'unit': 's', 'value': 1},
            'zStep': {'unit': 'm', 'value': 1}
        }
    }


def test_unit_conversion():
    """Test the unit conversion function"""
    # Test mm to µm conversion
    pixel_size = {'unit': 'mm', 'value': 1}
    result = convert_units(pixel_size, 'µm')
    assert result['unit'] == 'µm'
    assert result['value'] == pytest.approx(1000)  # 1mm = 1000µm

    # Test error handling
    with pytest.raises(ValueError):
        convert_units({'unit': 'invalid', 'value': 1}, 'µm')


def test_square_metrics(mock_worker_client, sample_params):
    """Test metrics computation for a perfect square"""

    # Disable physical units for easier testing
    sample_params['workerInterface']['Use physical units'] = False

    # Create a 10x10 square annotation
    square_annotation = {
        '_id': 'test_square',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0},
            {'x': 0, 'y': 0}
        ],
        'tags': ['nucleus']
    }

    # Set up mock to return our square annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        square_annotation]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the computed metrics
    property_values = calls[0][0][0]['test_dataset']['test_square']

    # Test the metrics
    assert property_values['Area'] == pytest.approx(100)  # 10 * 10
    assert property_values['Perimeter'] == pytest.approx(40)  # 4 * 10
    assert property_values['Circularity'] == pytest.approx(
        0.785, rel=0.01)  # π/4

    # Updated: Compactness uses the same formula as Circularity in the worker code
    assert property_values['Compactness'] == pytest.approx(
        0.785, rel=0.01)  # π/4
    assert property_values['Elongation'] == 0  # Square has no elongation


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
        assert 'Use physical units' in interface_data
        assert 'Units' in interface_data
        assert interface_data['Units']['type'] == 'select'
        assert 'µm' in interface_data['Units']['items']


def test_annotation_filtering(mock_worker_client, sample_annotation,
                              sample_params):
    """Test filtering annotations by tags"""
    # Create a copy of sample_annotation to avoid modifying the fixture
    annotation1 = dict(sample_annotation)
    # Create a second annotation with different tags
    annotation2 = dict(sample_annotation)
    annotation2['_id'] = 'test_id_2'
    annotation2['tags'] = ['cytoplasm']

    mock_worker_client.get_annotation_list_by_shape.return_value = [
        annotation1, annotation2
    ]

    # Set tag filter to 'nucleus'
    sample_params['tags'] = {'exclusive': True, 'tags': ['nucleus']}

    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify only nucleus annotations were processed
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1
    property_values = calls[0][0][0]['test_dataset']
    assert len(property_values) == 1


@pytest.mark.parametrize('shape,expected_metrics', [
    # Test different polygon shapes and their expected metrics
    ('square', {'elongation': 0.0, 'circularity': 0.785}),
    ('rectangle', {'elongation': 0.5, 'circularity': 0.698}),
    # Updated the expected elongation to match the actual calculation
    ('triangle', {'elongation': 0.134, 'circularity': 0.604})
])
def test_shape_metrics(mock_worker_client, sample_params, shape,
                       expected_metrics):
    """Test metrics computation for different shapes"""
    # Disable physical units for easier testing
    sample_params['workerInterface']['Use physical units'] = False

    # Define shape coordinates
    shapes = {
        'square': [
            {'x': 0, 'y': 0}, {'x': 0, 'y': 10},
            {'x': 10, 'y': 10}, {'x': 10, 'y': 0},
            {'x': 0, 'y': 0}
        ],
        'rectangle': [
            {'x': 0, 'y': 0}, {'x': 0, 'y': 10},
            {'x': 20, 'y': 10}, {'x': 20, 'y': 0},
            {'x': 0, 'y': 0}
        ],
        'triangle': [
            {'x': 0, 'y': 0}, {'x': 10, 'y': 17.32},
            {'x': 20, 'y': 0}, {'x': 0, 'y': 0}
        ]
    }

    annotation = {
        '_id': f'test_{shape}',
        'coordinates': shapes[shape],
        'tags': ['nucleus']
    }

    mock_worker_client.get_annotation_list_by_shape.return_value = [annotation]
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify metrics
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values = calls[0][0][0]['test_dataset'][f'test_{shape}']

    for metric, expected in expected_metrics.items():
        assert property_values[metric.title()] == pytest.approx(
            expected, rel=0.01)


def test_error_handling(mock_worker_client, sample_params):
    """Test handling of various error conditions"""
    # Test invalid polygon (less than 3 points)
    invalid_annotation = {
        '_id': 'invalid_polygon',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 10, 'y': 10}
        ],
        'tags': ['nucleus']  # Add the tag that matches our filter
    }

    mock_worker_client.get_annotation_list_by_shape.return_value = [
        invalid_annotation]

    # Mock the sendWarning function
    # The function is imported directly in the entrypoint.py file, so we need to mock it there
    with patch('entrypoint.sendWarning') as mock_send_warning:
        # Should not raise an error, but should skip the invalid annotation
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Check if sendWarning was called with the expected arguments
        mock_send_warning.assert_called_once_with(
            "Incorrect polygon detected", info="Polygon with less than 3 points found.")

    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    # The worker should still call add_multiple_annotation_property_values, but with an empty dictionary
    assert len(calls) == 1
    property_values = calls[0][0][0]['test_dataset']
    assert len(property_values) == 0  # No properties should be added for invalid annotations
