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
def mock_annotation_client():
    """Mock the UPennContrastAnnotationClient"""
    with patch(
        'annotation_client.annotations.UPennContrastAnnotationClient'
    ) as mock_client:
        client = mock_client.return_value
        # Set up default behaviors
        client.getAnnotationConnections.return_value = []
        client.getAnnotationsByDatasetId.return_value = []
        yield client


@pytest.fixture
def mock_annotation_tools():
    """Mock the annotation_tools module"""
    with patch(
        'annotation_utilities.annotation_tools.get_annotations_with_tags'
    ) as mock_get_annotations:
        mock_get_annotations.return_value = []
        yield mock_get_annotations


@pytest.fixture
def mock_pandas():
    """Mock pandas DataFrame operations"""
    with patch('pandas.DataFrame') as mock_df_class:
        # Create a mock DataFrame instance
        mock_df = MagicMock()
        mock_df_class.return_value = mock_df

        # Mock the groupby operation
        mock_groupby = MagicMock()
        mock_df.groupby.return_value = mock_groupby

        # Mock the size operation
        mock_size = MagicMock()
        mock_groupby.size.return_value = mock_size

        # Mock the reset_index operation
        mock_reset = MagicMock()
        mock_size.reset_index.return_value = mock_reset

        # Set up the result of the groupby operation
        mock_reset.__iter__.return_value = []
        mock_reset['parentId'] = []
        mock_reset['count'] = []

        yield mock_df_class


@pytest.fixture
def sample_annotation_t0():
    """Create a sample annotation at time 0"""
    return {
        '_id': 'annotation_1',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0},
            {'x': 0, 'y': 0}  # Close the polygon
        ],
        'tags': ['cell'],
        'location': {'Time': 0}
    }


@pytest.fixture
def sample_annotation_t1():
    """Create a sample annotation at time 1"""
    return {
        '_id': 'annotation_2',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 5},
            {'x': 5, 'y': 5}  # Close the polygon
        ],
        'tags': ['cell'],
        'location': {'Time': 1}
    }


@pytest.fixture
def sample_annotation_t2():
    """Create a sample annotation at time 2"""
    return {
        '_id': 'annotation_3',
        'coordinates': [
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 20},
            {'x': 20, 'y': 20},
            {'x': 20, 'y': 10},
            {'x': 10, 'y': 10}  # Close the polygon
        ],
        'tags': ['cell'],
        'location': {'Time': 2}
    }


@pytest.fixture
def sample_connection():
    """Create a sample connection between annotations"""
    return {
        'parentId': 'annotation_1',
        'childId': 'annotation_2',
        'dataset': 'test_dataset'
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'Parent Child',
        'image': 'properties/parent_child:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Ignore self-connections': True,
            'Time lapse': True,
            'Add track IDs': False
        }
    }


def test_interface():
    """Test the interface generation for parent child worker"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        # Verify interface was set
        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        # Verify interface structure
        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]

        # Check for Connection IDs notes
        assert 'Connection IDs' in interface_data
        assert interface_data['Connection IDs']['type'] == 'notes'

        # Check for Ignore self-connections checkbox
        assert 'Ignore self-connections' in interface_data
        assert interface_data['Ignore self-connections']['type'] == 'checkbox'
        assert interface_data['Ignore self-connections']['default'] is True

        # Check for Time lapse checkbox
        assert 'Time lapse' in interface_data
        assert interface_data['Time lapse']['type'] == 'checkbox'
        assert interface_data['Time lapse']['default'] is True

        # Check for Add track IDs checkbox
        assert 'Add track IDs' in interface_data
        assert interface_data['Add track IDs']['type'] == 'checkbox'
        assert interface_data['Add track IDs']['default'] is False


def test_worker_startup(mock_worker_client, mock_annotation_client, mock_annotation_tools, sample_params):
    """Test that the worker starts up correctly with no annotations"""
    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation with empty annotation list
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Verify that getAnnotationConnections was called
        mock_annotation_client.getAnnotationConnections.assert_called_once_with(
            'test_dataset', limit=10000000)

        # Verify that getAnnotationsByDatasetId was called
        mock_annotation_client.getAnnotationsByDatasetId.assert_called_once_with(
            'test_dataset', limit=10000000)

        # Verify that get_annotations_with_tags was called
        mock_annotation_tools.assert_called_once()

        # Since there are no annotations, add_multiple_annotation_property_values should be called with empty dict
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # The property values should be empty since there are no annotations
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values
        assert len(property_values['test_dataset']) == 0


def test_basic_connection(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                          sample_annotation_t0, sample_annotation_t1, sample_params):
    """Test basic connection between two annotations"""
    # Set up mock to return our annotations
    annotations = [sample_annotation_t0, sample_annotation_t1]
    mock_annotation_client.getAnnotationsByDatasetId.return_value = annotations
    mock_annotation_tools.return_value = annotations

    # Set up a connection between the annotations
    connection = {
        'parentId': 'annotation_1',
        'childId': 'annotation_2',
        'dataset': 'test_dataset'
    }
    mock_annotation_client.getAnnotationConnections.return_value = [connection]

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Verify that add_multiple_annotation_property_values was called
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # Get the property values that were sent to the server
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values

        # Check that the annotations have the correct parent/child IDs
        assert 'annotation_1' in property_values['test_dataset']
        assert 'annotation_2' in property_values['test_dataset']

        # Check that annotation_1 has a child and annotation_2 has a parent
        assert property_values['test_dataset']['annotation_1']['childId'] != -1.0
        assert property_values['test_dataset']['annotation_1']['parentId'] == -1.0

        assert property_values['test_dataset']['annotation_2']['parentId'] != -1.0
        assert property_values['test_dataset']['annotation_2']['childId'] == -1.0

        # Check that the parent-child relationship is correct
        # The integer IDs might not be 0 and 1, but they should match
        ann1_child_id = property_values['test_dataset']['annotation_1']['childId']
        ann2_id = property_values['test_dataset']['annotation_2']['annotationId']
        assert ann1_child_id == ann2_id

        ann2_parent_id = property_values['test_dataset']['annotation_2']['parentId']
        ann1_id = property_values['test_dataset']['annotation_1']['annotationId']
        assert ann2_parent_id == ann1_id


def test_time_lapse_connections(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                                sample_annotation_t0, sample_annotation_t1, sample_params):
    """Test time lapse connections where parent is earlier time and child is later time"""
    # Set up mock to return our annotations
    annotations = [sample_annotation_t0, sample_annotation_t1]
    mock_annotation_client.getAnnotationsByDatasetId.return_value = annotations
    mock_annotation_tools.return_value = annotations

    # Set up a connection where the child is at an earlier time than the parent
    # This should be reversed by the time lapse logic
    connection = {
        'parentId': 'annotation_2',  # Time 1
        'childId': 'annotation_1',   # Time 0
        'dataset': 'test_dataset'
    }
    mock_annotation_client.getAnnotationConnections.return_value = [connection]

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation with time lapse enabled
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Verify that add_multiple_annotation_property_values was called
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # Get the property values that were sent to the server
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values

        # Check that the annotations have the correct parent/child IDs
        # The connection should be reversed because of time lapse
        assert 'annotation_1' in property_values['test_dataset']
        assert 'annotation_2' in property_values['test_dataset']

        # Check that annotation_1 has a child and annotation_2 has a parent
        assert property_values['test_dataset']['annotation_1']['childId'] != -1.0
        assert property_values['test_dataset']['annotation_1']['parentId'] == -1.0

        assert property_values['test_dataset']['annotation_2']['parentId'] != -1.0
        assert property_values['test_dataset']['annotation_2']['childId'] == -1.0

        # Check that the parent-child relationship is correct
        # The integer IDs might not be 0 and 1, but they should match
        ann1_child_id = property_values['test_dataset']['annotation_1']['childId']
        ann2_id = property_values['test_dataset']['annotation_2']['annotationId']
        assert ann1_child_id == ann2_id

        ann2_parent_id = property_values['test_dataset']['annotation_2']['parentId']
        ann1_id = property_values['test_dataset']['annotation_1']['annotationId']
        assert ann2_parent_id == ann1_id


def test_no_time_lapse(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                       sample_annotation_t0, sample_annotation_t1):
    """Test connections when time lapse is disabled"""
    # Set up mock to return our annotations
    annotations = [sample_annotation_t0, sample_annotation_t1]
    mock_annotation_client.getAnnotationsByDatasetId.return_value = annotations
    mock_annotation_tools.return_value = annotations

    # Set up a connection where the child is at an earlier time than the parent
    connection = {
        'parentId': 'annotation_2',  # Time 1
        'childId': 'annotation_1',   # Time 0
        'dataset': 'test_dataset'
    }
    mock_annotation_client.getAnnotationConnections.return_value = [connection]

    # Create parameters with time lapse disabled
    params = {
        'id': 'test_property_id',
        'name': 'Parent Child',
        'image': 'properties/parent_child:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Ignore self-connections': True,
            'Time lapse': False,  # Disable time lapse
            'Add track IDs': False
        }
    }

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation with time lapse disabled
        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify that add_multiple_annotation_property_values was called
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # Get the property values that were sent to the server
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values

        # Check that the annotations have the correct parent/child IDs
        # The connection should NOT be reversed because time lapse is disabled
        assert 'annotation_1' in property_values['test_dataset']
        assert 'annotation_2' in property_values['test_dataset']

        # Check that annotation_1 has a parent and annotation_2 has a child
        assert property_values['test_dataset']['annotation_1']['childId'] == -1.0
        assert property_values['test_dataset']['annotation_1']['parentId'] != -1.0

        assert property_values['test_dataset']['annotation_2']['parentId'] == -1.0
        assert property_values['test_dataset']['annotation_2']['childId'] != -1.0

        # Check that the parent-child relationship is correct
        # The integer IDs might not be 0 and 1, but they should match
        ann1_parent_id = property_values['test_dataset']['annotation_1']['parentId']
        ann2_id = property_values['test_dataset']['annotation_2']['annotationId']
        assert ann1_parent_id == ann2_id

        ann2_child_id = property_values['test_dataset']['annotation_2']['childId']
        ann1_id = property_values['test_dataset']['annotation_1']['annotationId']
        assert ann2_child_id == ann1_id


def test_self_connection(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                         sample_annotation_t0, sample_params):
    """Test handling of self-connections"""
    # Set up mock to return our annotation
    mock_annotation_client.getAnnotationsByDatasetId.return_value = [sample_annotation_t0]
    mock_annotation_tools.return_value = [sample_annotation_t0]

    # Set up a self-connection
    connection = {
        'parentId': 'annotation_1',
        'childId': 'annotation_1',
        'dataset': 'test_dataset'
    }
    mock_annotation_client.getAnnotationConnections.return_value = [connection]

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation with ignore self-connections enabled
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Verify that add_multiple_annotation_property_values was called
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # Get the property values that were sent to the server
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values

        # Check that the annotation has no parent or child (self-connection ignored)
        assert 'annotation_1' in property_values['test_dataset']
        assert property_values['test_dataset']['annotation_1']['parentId'] == -1.0
        assert property_values['test_dataset']['annotation_1']['childId'] == -1.0


def test_allow_self_connection(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                               sample_annotation_t0):
    """Test allowing self-connections"""
    # Set up mock to return our annotation
    mock_annotation_client.getAnnotationsByDatasetId.return_value = [sample_annotation_t0]
    mock_annotation_tools.return_value = [sample_annotation_t0]

    # Set up a self-connection
    connection = {
        'parentId': 'annotation_1',
        'childId': 'annotation_1',
        'dataset': 'test_dataset'
    }
    mock_annotation_client.getAnnotationConnections.return_value = [connection]

    # Create parameters with ignore self-connections disabled
    params = {
        'id': 'test_property_id',
        'name': 'Parent Child',
        'image': 'properties/parent_child:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Ignore self-connections': False,  # Allow self-connections
            'Time lapse': True,
            'Add track IDs': False
        }
    }

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation with ignore self-connections disabled
        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify that add_multiple_annotation_property_values was called
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # Get the property values that were sent to the server
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values

        # Check that the annotation has itself as both parent and child
        assert 'annotation_1' in property_values['test_dataset']

        # Get the annotation's integer ID
        ann1_id = property_values['test_dataset']['annotation_1']['annotationId']

        # Check that the annotation has itself as both parent and child
        assert property_values['test_dataset']['annotation_1']['parentId'] == ann1_id
        assert property_values['test_dataset']['annotation_1']['childId'] == ann1_id


def test_track_ids(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                   sample_annotation_t0, sample_annotation_t1, sample_annotation_t2):
    """Test track ID assignment for connected annotations"""
    # Set up mock to return our annotations
    annotations = [sample_annotation_t0, sample_annotation_t1, sample_annotation_t2]
    mock_annotation_client.getAnnotationsByDatasetId.return_value = annotations
    mock_annotation_tools.return_value = annotations

    # Set up connections to form a track: annotation_1 -> annotation_2 -> annotation_3
    connections = [
        {
            'parentId': 'annotation_1',
            'childId': 'annotation_2',
            'dataset': 'test_dataset'
        },
        {
            'parentId': 'annotation_2',
            'childId': 'annotation_3',
            'dataset': 'test_dataset'
        }
    ]
    mock_annotation_client.getAnnotationConnections.return_value = connections

    # Create parameters with track IDs enabled
    params = {
        'id': 'test_property_id',
        'name': 'Parent Child',
        'image': 'properties/parent_child:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Ignore self-connections': True,
            'Time lapse': True,
            'Add track IDs': True  # Enable track IDs
        }
    }

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation with track IDs enabled
        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify that add_multiple_annotation_property_values was called
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # Get the property values that were sent to the server
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values

        # Check that all annotations have the same track ID
        assert 'annotation_1' in property_values['test_dataset']
        assert 'annotation_2' in property_values['test_dataset']
        assert 'annotation_3' in property_values['test_dataset']

        track_id_1 = property_values['test_dataset']['annotation_1']['trackId']
        track_id_2 = property_values['test_dataset']['annotation_2']['trackId']
        track_id_3 = property_values['test_dataset']['annotation_3']['trackId']

        assert track_id_1 == track_id_2 == track_id_3

        # Also check that parent/child relationships are correct
        # Get the annotation IDs
        ann1_id = property_values['test_dataset']['annotation_1']['annotationId']
        ann2_id = property_values['test_dataset']['annotation_2']['annotationId']
        ann3_id = property_values['test_dataset']['annotation_3']['annotationId']

        # Check parent-child relationships
        assert property_values['test_dataset']['annotation_1']['childId'] == ann2_id
        assert property_values['test_dataset']['annotation_1']['parentId'] == -1.0

        assert property_values['test_dataset']['annotation_2']['parentId'] == ann1_id
        assert property_values['test_dataset']['annotation_2']['childId'] == ann3_id

        assert property_values['test_dataset']['annotation_3']['parentId'] == ann2_id
        assert property_values['test_dataset']['annotation_3']['childId'] == -1.0


def test_multiple_tracks(mock_worker_client, mock_annotation_client, mock_annotation_tools,
                         sample_annotation_t0, sample_annotation_t1, sample_annotation_t2):
    """Test multiple separate tracks with different track IDs"""
    # Set up mock to return our annotations plus two more for a separate track
    annotation_4 = {
        '_id': 'annotation_4',
        'coordinates': [{'x': 30, 'y': 30}, {'x': 30, 'y': 40}, {'x': 40, 'y': 40},
                        {'x': 40, 'y': 30}, {'x': 30, 'y': 30}],
        'tags': ['cell'],
        'location': {'Time': 0}
    }

    annotation_5 = {
        '_id': 'annotation_5',
        'coordinates': [{'x': 35, 'y': 35}, {'x': 35, 'y': 45}, {'x': 45, 'y': 45},
                        {'x': 45, 'y': 35}, {'x': 35, 'y': 35}],
        'tags': ['cell'],
        'location': {'Time': 1}
    }

    annotations = [sample_annotation_t0, sample_annotation_t1,
                   sample_annotation_t2, annotation_4, annotation_5]
    mock_annotation_client.getAnnotationsByDatasetId.return_value = annotations
    mock_annotation_tools.return_value = annotations

    # Set up connections to form two separate tracks:
    # Track 1: annotation_1 -> annotation_2 -> annotation_3
    # Track 2: annotation_4 -> annotation_5
    connections = [
        {
            'parentId': 'annotation_1',
            'childId': 'annotation_2',
            'dataset': 'test_dataset'
        },
        {
            'parentId': 'annotation_2',
            'childId': 'annotation_3',
            'dataset': 'test_dataset'
        },
        {
            'parentId': 'annotation_4',
            'childId': 'annotation_5',
            'dataset': 'test_dataset'
        }
    ]
    mock_annotation_client.getAnnotationConnections.return_value = connections

    # Create parameters with track IDs enabled
    params = {
        'id': 'test_property_id',
        'name': 'Parent Child',
        'image': 'properties/parent_child:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Ignore self-connections': True,
            'Time lapse': True,
            'Add track IDs': True  # Enable track IDs
        }
    }

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation with track IDs enabled
        compute('test_dataset', 'http://test-api', 'test-token', params)

        # Verify that add_multiple_annotation_property_values was called
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # Get the property values that were sent to the server
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values

        # Check that annotations in the same track have the same track ID
        track_id_1 = property_values['test_dataset']['annotation_1']['trackId']
        track_id_2 = property_values['test_dataset']['annotation_2']['trackId']
        track_id_3 = property_values['test_dataset']['annotation_3']['trackId']

        track_id_4 = property_values['test_dataset']['annotation_4']['trackId']
        track_id_5 = property_values['test_dataset']['annotation_5']['trackId']

        # Annotations 1, 2, 3 should have the same track ID
        assert track_id_1 == track_id_2 == track_id_3

        # Annotations 4, 5 should have the same track ID
        assert track_id_4 == track_id_5

        # The two tracks should have different track IDs
        assert track_id_1 != track_id_4
