import pytest
from unittest.mock import patch, MagicMock
import math

# Import worker module
from entrypoint import compute, interface, calculate_distance_to_blob


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
def mock_annotation_client():
    """Mock the UPennContrastAnnotationClient"""
    with patch(
        'annotation_client.annotations.UPennContrastAnnotationClient'
    ) as mock_client:
        client = mock_client.return_value
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
            {'x': 100.0, 'y': 200.0}
        ],
        'tags': ['nucleus'],
        'location': 'test_location'
    }


@pytest.fixture
def sample_blob_annotation():
    """Create a sample blob annotation"""
    return {
        '_id': 'test_blob_1',
        'coordinates': [
            {'x': 90.0, 'y': 190.0},
            {'x': 110.0, 'y': 190.0},
            {'x': 110.0, 'y': 210.0},
            {'x': 90.0, 'y': 210.0}
        ],
        'tags': ['cell'],
        'location': 'test_location'
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'Point to Nearest Blob Distance',
        'image': 'properties/point_to_nearest_blob_distance:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'point',
        'workerInterface': {
            'Blob tags': ['cell'],
            'Distance type': 'Centroid',
            'Create connection': False
        }
    }


def test_interface():
    """Test the interface generation for point to nearest blob distance"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        # Verify interface was set
        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        # Verify interface structure
        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]
        
        # Check required fields
        assert 'Blob tags' in interface_data
        assert interface_data['Blob tags']['type'] == 'tags'
        assert interface_data['Blob tags']['required'] == True
        
        assert 'Distance type' in interface_data
        assert interface_data['Distance type']['type'] == 'select'
        assert 'Centroid' in interface_data['Distance type']['items']
        assert 'Edge' in interface_data['Distance type']['items']
        assert interface_data['Distance type']['default'] == 'Centroid'
        
        assert 'Create connection' in interface_data
        assert interface_data['Create connection']['type'] == 'checkbox'
        assert interface_data['Create connection']['default'] == False


def test_calculate_distance_to_blob_centroid():
    """Test distance calculation to blob centroid"""
    point = {
        'coordinates': [{'x': 0.0, 'y': 0.0}]
    }
    
    # Square blob centered at (10, 10) with side length 2
    blob = {
        'coordinates': [
            {'x': 9.0, 'y': 9.0},
            {'x': 11.0, 'y': 9.0},
            {'x': 11.0, 'y': 11.0},
            {'x': 9.0, 'y': 11.0}
        ]
    }
    
    distance = calculate_distance_to_blob(point, blob, 'centroid')
    
    # Distance from (0,0) to centroid (10,10) should be sqrt(200) ≈ 14.14
    expected_distance = math.sqrt(200)
    assert abs(distance - expected_distance) < 0.01


def test_calculate_distance_to_blob_edge():
    """Test distance calculation to blob edge"""
    point = {
        'coordinates': [{'x': 0.0, 'y': 10.0}]
    }
    
    # Square blob from (9,9) to (11,11)
    blob = {
        'coordinates': [
            {'x': 9.0, 'y': 9.0},
            {'x': 11.0, 'y': 9.0},
            {'x': 11.0, 'y': 11.0},
            {'x': 9.0, 'y': 11.0}
        ]
    }
    
    distance = calculate_distance_to_blob(point, blob, 'edge')
    
    # Distance from (0,10) to nearest edge should be 9.0
    assert abs(distance - 9.0) < 0.01


def test_calculate_distance_invalid_type():
    """Test distance calculation with invalid distance type"""
    point = {
        'coordinates': [{'x': 0.0, 'y': 0.0}]
    }
    
    blob = {
        'coordinates': [
            {'x': 9.0, 'y': 9.0},
            {'x': 11.0, 'y': 9.0},
            {'x': 11.0, 'y': 11.0},
            {'x': 9.0, 'y': 11.0}
        ]
    }
    
    with pytest.raises(ValueError, match="Invalid distance_type"):
        calculate_distance_to_blob(point, blob, 'invalid')


def test_basic_compute(mock_worker_client, mock_annotation_client, mock_annotation_tools, 
                      sample_point_annotation, sample_blob_annotation, sample_params):
    """Test basic computation with one point and one blob"""
    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_point_annotation],  # points
        [sample_blob_annotation]    # blobs
    ]
    mock_annotation_tools.side_effect = [
        [sample_point_annotation],  # filtered points
        [sample_blob_annotation]    # filtered blobs
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify point retrieval
    calls = mock_worker_client.get_annotation_list_by_shape.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == 'point'
    assert calls[1][0][0] == 'polygon'

    # Verify property values were added
    mock_worker_client.add_multiple_annotation_property_values.assert_called_once()
    
    # Get the property values that were sent
    property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]['test_dataset']
    
    # Verify distance was calculated and stored
    assert 'test_point_1' in property_values
    assert isinstance(property_values['test_point_1'], float)
    # Point is at (100, 200) and blob centroid is at (100, 200), so distance should be 0.0
    assert property_values['test_point_1'] == 0.0


def test_compute_with_connection_creation(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                                        sample_point_annotation, sample_blob_annotation, sample_params):
    """Test computation with connection creation enabled"""
    # Enable connection creation
    sample_params['workerInterface']['Create connection'] = True
    
    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_point_annotation],
        [sample_blob_annotation]
    ]
    mock_annotation_tools.side_effect = [
        [sample_point_annotation],
        [sample_blob_annotation]
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify connections were created
    mock_annotation_client.createMultipleConnections.assert_called_once()
    
    # Check connection structure
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]
    assert len(connections) == 1
    
    connection = connections[0]
    assert connection['datasetId'] == 'test_dataset'
    assert connection['parentId'] == 'test_blob_1'
    assert connection['childId'] == 'test_point_1'
    assert 'nucleus' in connection['tags']
    assert 'cell' in connection['tags']


def test_compute_different_locations(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                                   sample_point_annotation, sample_blob_annotation, sample_params):
    """Test that points and blobs in different locations are not matched"""
    # Put point and blob in different locations
    sample_point_annotation['location'] = 'location_1'
    sample_blob_annotation['location'] = 'location_2'
    
    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_point_annotation],
        [sample_blob_annotation]
    ]
    mock_annotation_tools.side_effect = [
        [sample_point_annotation],
        [sample_blob_annotation]
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify no property values were added (no matching locations)
    property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]['test_dataset']
    assert len(property_values) == 0


def test_compute_multiple_blobs_finds_nearest(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                                            sample_point_annotation, sample_params):
    """Test that the nearest blob is selected when multiple blobs exist"""
    # Create two blobs at different distances
    near_blob = {
        '_id': 'near_blob',
        'coordinates': [
            {'x': 95.0, 'y': 195.0},
            {'x': 105.0, 'y': 195.0},
            {'x': 105.0, 'y': 205.0},
            {'x': 95.0, 'y': 205.0}
        ],
        'tags': ['cell'],
        'location': 'test_location'
    }
    
    far_blob = {
        '_id': 'far_blob',
        'coordinates': [
            {'x': 200.0, 'y': 300.0},
            {'x': 210.0, 'y': 300.0},
            {'x': 210.0, 'y': 310.0},
            {'x': 200.0, 'y': 310.0}
        ],
        'tags': ['cell'],
        'location': 'test_location'
    }
    
    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_point_annotation],
        [near_blob, far_blob]
    ]
    mock_annotation_tools.side_effect = [
        [sample_point_annotation],
        [near_blob, far_blob]
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify property was added
    property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]['test_dataset']
    assert 'test_point_1' in property_values
    
    # Distance to near blob should be much smaller than to far blob
    # (Point at 100,200, near blob centroid at 100,200, far blob centroid at 205,305)
    distance = property_values['test_point_1']
    assert distance < 10  # Should be very close to near blob


def test_compute_edge_distance_type(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                                  sample_point_annotation, sample_blob_annotation, sample_params):
    """Test computation with edge distance type"""
    # Change distance type to edge
    sample_params['workerInterface']['Distance type'] = 'Edge'
    
    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_point_annotation],
        [sample_blob_annotation]
    ]
    mock_annotation_tools.side_effect = [
        [sample_point_annotation],
        [sample_blob_annotation]
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify property values were added
    property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]['test_dataset']
    assert 'test_point_1' in property_values
    assert isinstance(property_values['test_point_1'], float)


def test_compute_no_points(mock_worker_client, mock_annotation_client, mock_annotation_tools, sample_params):
    """Test handling when no point annotations exist"""
    # Set up mocks to return empty lists
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [],  # no points
        []   # no blobs
    ]
    mock_annotation_tools.side_effect = [
        [],  # no filtered points
        []   # no filtered blobs
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify add_multiple_annotation_property_values was still called with empty dict
    mock_worker_client.add_multiple_annotation_property_values.assert_called_once()
    property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]['test_dataset']
    assert len(property_values) == 0


def test_compute_no_blobs(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                         sample_point_annotation, sample_params):
    """Test handling when no blob annotations exist"""
    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_point_annotation],  # points exist
        []                          # no blobs
    ]
    mock_annotation_tools.side_effect = [
        [sample_point_annotation],  # filtered points
        []                          # no filtered blobs
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify no property values were added (no blobs to measure distance to)
    property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]['test_dataset']
    assert len(property_values) == 0


def test_compute_no_matching_blob_tags(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                                     sample_point_annotation, sample_blob_annotation, sample_params):
    """Test handling when blob tags don't match filter"""
    # Change blob tags to something that won't match
    sample_params['workerInterface']['Blob tags'] = ['other_tag']
    
    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_point_annotation],
        [sample_blob_annotation]
    ]
    mock_annotation_tools.side_effect = [
        [sample_point_annotation],  # filtered points
        []                          # no filtered blobs due to tag mismatch
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify no property values were added
    property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]['test_dataset']
    assert len(property_values) == 0


def test_multiple_points_different_distances(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                                           sample_blob_annotation, sample_params):
    """Test multiple points at different distances from blob"""
    # Create points at different distances
    close_point = {
        '_id': 'close_point',
        'coordinates': [{'x': 95.0, 'y': 195.0}],
        'tags': ['nucleus'],
        'location': 'test_location'
    }
    
    far_point = {
        '_id': 'far_point',
        'coordinates': [{'x': 150.0, 'y': 250.0}],
        'tags': ['nucleus'],
        'location': 'test_location'
    }
    
    # Set up mocks
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [close_point, far_point],
        [sample_blob_annotation]
    ]
    mock_annotation_tools.side_effect = [
        [close_point, far_point],
        [sample_blob_annotation]
    ]

    # Run computation
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify both points got distance values
    property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]['test_dataset']
    
    assert 'close_point' in property_values
    assert 'far_point' in property_values
    
    # Verify the close point has a smaller distance
    assert property_values['close_point'] < property_values['far_point'] 