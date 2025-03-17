import pytest
from unittest.mock import patch, MagicMock

# Import worker module
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
def mock_annotation_tools():
    """Mock the annotation_tools module"""
    with patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_annotations:
        # Set up default behaviors
        mock_get_annotations.return_value = []
        yield mock_get_annotations


@pytest.fixture
def mock_send_progress():
    """Mock the sendProgress function"""
    with patch('annotation_client.utils.sendProgress') as mock_progress:
        yield mock_progress


@pytest.fixture
def sample_point_annotation():
    """Create a sample point annotation"""
    return {
        '_id': 'test_point_1',
        'coordinates': [
            {'x': 100, 'y': 200}
        ],
        'tags': ['nucleus']  # Add the tag that matches our filter
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'Point Coordinates',
        'image': 'properties/point_metrics:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'point'
    }


def test_interface():
    """Test the interface generation for point metrics"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        # Verify interface was set
        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        # Verify interface structure
        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]
        assert 'Point Metrics' in interface_data
        assert interface_data['Point Metrics']['type'] == 'notes'
        assert 'coordinates' in interface_data['Point Metrics']['value']


def test_annotation_filtering(mock_worker_client, mock_annotation_tools, sample_point_annotation, sample_params):
    """Test filtering annotations by tags"""
    # Create a second point with different tags
    point2 = dict(sample_point_annotation)
    point2['_id'] = 'test_point_2'
    point2['tags'] = ['cytoplasm']

    # Set up mock to return both annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_point_annotation, point2
    ]

    # Set up mock to filter and only return the nucleus annotation
    mock_annotation_tools.return_value = [sample_point_annotation]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify tag filtering was called with correct parameters
    mock_annotation_tools.assert_called_with(
        [sample_point_annotation, point2],
        sample_params.get('tags', {}).get('tags', []),
        sample_params.get('tags', {}).get('exclusive', False)
    )

    # Verify property values were added for the filtered annotation
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the property values that were sent to the server
    property_values = calls[0][0][0]['test_dataset']

    # Verify only one annotation was processed (the one with 'nucleus' tag)
    assert len(property_values) == 1
    assert 'test_point_1' in property_values

    # Verify the point coordinates were correctly stored
    assert property_values['test_point_1']['x'] == 100
    assert property_values['test_point_1']['y'] == 200


def test_no_annotations(mock_worker_client, mock_annotation_tools, sample_params):
    """Test handling of no annotations"""
    # Set up mock to return empty list
    mock_worker_client.get_annotation_list_by_shape.return_value = []
    mock_annotation_tools.return_value = []

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify get_annotation_list_by_shape was called with correct parameters
    mock_worker_client.get_annotation_list_by_shape.assert_called_with('point', limit=0)

    # Verify add_multiple_annotation_property_values was not called
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_empty_tag_list(mock_worker_client, mock_annotation_tools, sample_point_annotation):
    """Test handling of empty tag list in parameters"""
    # Create params with empty tag list
    params_empty_tags = {
        'id': 'test_property_id',
        'name': 'Point Coordinates',
        'image': 'properties/point_metrics:latest',
        'tags': {'exclusive': False, 'tags': []},
        'shape': 'point'
    }

    # Set up mock to return our point annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_point_annotation]
    mock_annotation_tools.return_value = [sample_point_annotation]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', params_empty_tags)

    # Verify tag filtering was called with empty tag list
    mock_annotation_tools.assert_called_with(
        [sample_point_annotation],
        [],
        False
    )

    # Verify property values were added
    mock_worker_client.add_multiple_annotation_property_values.assert_called_once()


def test_no_tags_parameter(mock_worker_client, mock_annotation_tools, sample_point_annotation):
    """Test handling of missing tags parameter"""
    # Create params without tags parameter
    params_no_tags = {
        'id': 'test_property_id',
        'name': 'Point Coordinates',
        'image': 'properties/point_metrics:latest',
        'shape': 'point'
    }

    # Set up mock to return our point annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_point_annotation]
    mock_annotation_tools.return_value = [sample_point_annotation]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', params_no_tags)

    # Verify tag filtering was called with empty tag list
    mock_annotation_tools.assert_called_with(
        [sample_point_annotation],
        [],
        False
    )

    # Verify property values were added
    mock_worker_client.add_multiple_annotation_property_values.assert_called_once()


def test_multiple_points(mock_worker_client, mock_annotation_tools, sample_point_annotation):
    """Test processing of multiple point annotations"""
    # Create multiple point annotations
    point1 = dict(sample_point_annotation)
    point2 = dict(sample_point_annotation)
    point2['_id'] = 'test_point_2'
    point2['coordinates'] = [{'x': 300, 'y': 400}]
    point3 = dict(sample_point_annotation)
    point3['_id'] = 'test_point_3'
    point3['coordinates'] = [{'x': 500, 'y': 600}]

    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.return_value = [point1, point2, point3]
    mock_annotation_tools.return_value = [point1, point2, point3]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', {})

    # Verify property values were added for all points
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    assert len(calls) == 1

    # Get the property values that were sent to the server
    property_values = calls[0][0][0]['test_dataset']

    # Verify all three points were processed
    assert len(property_values) == 3
    assert 'test_point_1' in property_values
    assert 'test_point_2' in property_values
    assert 'test_point_3' in property_values

    # Verify the coordinates were correctly stored
    assert property_values['test_point_1']['x'] == 100
    assert property_values['test_point_1']['y'] == 200
    assert property_values['test_point_2']['x'] == 300
    assert property_values['test_point_2']['y'] == 400
    assert property_values['test_point_3']['x'] == 500
    assert property_values['test_point_3']['y'] == 600


def test_progress_reporting(mock_worker_client, mock_annotation_tools, sample_params, capsys):
    """Test progress reporting for different numbers of annotations"""
    # Create a list of 10 point annotations
    annotations = []
    for i in range(10):
        annotations.append({
            '_id': f'test_point_{i}',
            'coordinates': [{'x': i*10, 'y': i*20}],
            'tags': ['nucleus']
        })

    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.return_value = annotations
    mock_annotation_tools.return_value = annotations

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Capture the stdout output
    captured = capsys.readouterr()

    # Verify progress messages were printed for each annotation
    for i in range(1, 11):
        assert f'"progress": {i/10}' in captured.out
        assert f'"info": "Processing annotation {i}/10"' in captured.out

    # Verify the final progress message
    assert '"title": "Done computing"' in captured.out
    assert '"info": "Sending computed metrics to the server"' in captured.out

    # Verify property values were added
    mock_worker_client.add_multiple_annotation_property_values.assert_called_once()
