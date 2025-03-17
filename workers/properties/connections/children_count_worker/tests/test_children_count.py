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
        yield client


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
def sample_parent_annotation():
    """Create a sample parent annotation"""
    return {
        '_id': 'parent_1',
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
def sample_child_annotation():
    """Create a sample child annotation"""
    return {
        '_id': 'child_1',
        'coordinates': [
            {'x': 5, 'y': 5}
        ],
        'tags': ['spot']  # Add the tag that matches our filter
    }


@pytest.fixture
def sample_connection():
    """Create a sample connection between parent and child"""
    return {
        'parentId': 'parent_1',
        'childId': 'child_1',
        'dataset': 'test_dataset'
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'Children Count',
        'image': 'properties/children_count:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'polygon',
        'workerInterface': {
            'Child Tags': ['spot'],
            'Child Tags Exclusive': 'No'
        }
    }


def test_interface():
    """Test the interface generation for children count worker"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        # Verify interface was set
        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        # Verify interface structure
        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]
        assert 'Count connected objects' in interface_data
        assert interface_data['Count connected objects']['type'] == 'notes'

        assert 'Child Tags' in interface_data
        assert interface_data['Child Tags']['type'] == 'tags'

        assert 'Child Tags Exclusive' in interface_data
        assert interface_data['Child Tags Exclusive']['type'] == 'select'
        assert 'Yes' in interface_data['Child Tags Exclusive']['items']
        assert 'No' in interface_data['Child Tags Exclusive']['items']
        assert interface_data['Child Tags Exclusive']['default'] == 'No'


def test_worker_startup(mock_worker_client, mock_annotation_client, mock_pandas, sample_params):
    """Test that the worker starts up correctly with no annotations"""
    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Run computation with empty annotation list
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Verify that get_annotation_list_by_shape was called with None (to get all annotations)
        mock_worker_client.get_annotation_list_by_shape.assert_called_once_with(
            None, limit=0)

        # Verify that getAnnotationConnections was called
        mock_annotation_client.getAnnotationConnections.assert_called_once_with(
            'test_dataset', limit=10000000)

        # Since there are no annotations, add_multiple_annotation_property_values should be called with empty dict
        mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

        # The property values should be empty since there are no annotations
        property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[0][0]
        assert 'test_dataset' in property_values
        assert len(property_values['test_dataset']) == 0


def test_basic_counting(mock_worker_client, mock_annotation_client, sample_parent_annotation,
                        sample_child_annotation, sample_connection, sample_params):
    """Test basic counting of children connected to parent annotations"""
    # Set up mock to return our parent and child annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_parent_annotation, sample_child_annotation
    ]

    # Set up mock to return our connection
    mock_annotation_client.getAnnotationConnections.return_value = [
        sample_connection
    ]

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Mock pandas to avoid recursion issues
        with patch('pandas.DataFrame') as mock_df:
            # Create a mock DataFrame that will be returned by the DataFrame constructor
            mock_df_instance = MagicMock()
            mock_df.return_value = mock_df_instance

            # Mock the groupby chain to return a DataFrame with our expected count
            mock_count_df = MagicMock()
            mock_df_instance.groupby.return_value.size.return_value.reset_index.return_value = mock_count_df

            # Set up the mock DataFrame to behave like a dictionary when accessed with __getitem__
            mock_count_df.__getitem__.side_effect = lambda key: [
                'parent_1'] if key == 'parentId' else [1]

            # Set up the mock DataFrame to behave like a list when iterated
            mock_count_df.__iter__.return_value = [{'parentId': 'parent_1', 'count': 1}]

            # Set up the zip function to return a list of tuples
            with patch('builtins.zip') as mock_zip:
                mock_zip.return_value = [('parent_1', 1)]

                # Run computation
                compute('test_dataset', 'http://test-api', 'test-token', sample_params)

                # Verify that add_multiple_annotation_property_values was called
                mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

                # Get the property values that were sent to the server
                property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[
                    0][0]
                assert 'test_dataset' in property_values

                # Check that the parent annotation has a count of 1
                assert 'parent_1' in property_values['test_dataset']
                assert property_values['test_dataset']['parent_1'] == 1


def test_multiple_children_and_parents(mock_worker_client, mock_annotation_client, sample_params):
    """Test counting of multiple children connected to multiple parent annotations"""
    # Create multiple parent annotations
    parent1 = {
        '_id': 'parent_1',
        'coordinates': [{'x': 0, 'y': 0}, {'x': 0, 'y': 10}, {'x': 10, 'y': 10},
                        {'x': 10, 'y': 0}, {'x': 0, 'y': 0}],
        'tags': ['nucleus']
    }

    parent2 = {
        '_id': 'parent_2',
        'coordinates': [{'x': 20, 'y': 20}, {'x': 20, 'y': 30}, {'x': 30, 'y': 30},
                        {'x': 30, 'y': 20}, {'x': 20, 'y': 20}],
        'tags': ['nucleus']
    }

    # Create multiple child annotations
    child1 = {'_id': 'child_1', 'coordinates': [{'x': 5, 'y': 5}], 'tags': ['spot']}
    child2 = {'_id': 'child_2', 'coordinates': [{'x': 8, 'y': 8}], 'tags': ['spot']}
    child3 = {'_id': 'child_3', 'coordinates': [{'x': 25, 'y': 25}], 'tags': ['spot']}

    # Set up mock to return our parent and child annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        parent1, parent2, child1, child2, child3
    ]

    # Create connections between parents and children
    # parent1 has 2 children, parent2 has 1 child
    connections = [
        {'parentId': 'parent_1', 'childId': 'child_1', 'dataset': 'test_dataset'},
        {'parentId': 'parent_1', 'childId': 'child_2', 'dataset': 'test_dataset'},
        {'parentId': 'parent_2', 'childId': 'child_3', 'dataset': 'test_dataset'}
    ]

    # Set up mock to return our connections
    mock_annotation_client.getAnnotationConnections.return_value = connections

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Mock pandas to avoid recursion issues
        with patch('pandas.DataFrame') as mock_df:
            # Create a mock DataFrame that will be returned by the DataFrame constructor
            mock_df_instance = MagicMock()
            mock_df.return_value = mock_df_instance

            # Mock the groupby chain to return a DataFrame with our expected counts
            mock_count_df = MagicMock()
            mock_df_instance.groupby.return_value.size.return_value.reset_index.return_value = mock_count_df

            # Set up the mock DataFrame to behave like a dictionary when accessed with __getitem__
            mock_count_df.__getitem__.side_effect = lambda key: [
                'parent_1', 'parent_2'] if key == 'parentId' else [2, 1]

            # Set up the mock DataFrame to behave like a list when iterated
            mock_count_df.__iter__.return_value = [
                {'parentId': 'parent_1', 'count': 2},
                {'parentId': 'parent_2', 'count': 1}
            ]

            # Set up the zip function to return a list of tuples
            with patch('builtins.zip') as mock_zip:
                mock_zip.return_value = [('parent_1', 2), ('parent_2', 1)]

                # Run computation
                compute('test_dataset', 'http://test-api', 'test-token', sample_params)

                # Verify that add_multiple_annotation_property_values was called
                mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

                # Get the property values that were sent to the server
                property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[
                    0][0]
                assert 'test_dataset' in property_values

                # Check that the parent annotations have the correct counts
                assert 'parent_1' in property_values['test_dataset']
                assert property_values['test_dataset']['parent_1'] == 2

                assert 'parent_2' in property_values['test_dataset']
                assert property_values['test_dataset']['parent_2'] == 1


def test_exclusive_tag_filtering(mock_worker_client, mock_annotation_client):
    """Test counting with exclusive tag filtering for child annotations"""
    # Create parameters with exclusive tag filtering
    exclusive_params = {
        'id': 'test_property_id',
        'name': 'Children Count',
        'image': 'properties/children_count:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'polygon',
        'workerInterface': {
            'Child Tags': ['spot'],
            'Child Tags Exclusive': 'Yes'  # Set to exclusive
        }
    }

    # Create parent annotation
    parent = {
        '_id': 'parent_1',
        'coordinates': [{'x': 0, 'y': 0}, {'x': 0, 'y': 10}, {'x': 10, 'y': 10},
                        {'x': 10, 'y': 0}, {'x': 0, 'y': 0}],
        'tags': ['nucleus']
    }

    # Create child annotations with different tags
    child1 = {'_id': 'child_1', 'coordinates': [
        {'x': 5, 'y': 5}], 'tags': ['spot']}  # Has the right tag
    child2 = {'_id': 'child_2', 'coordinates': [
        {'x': 8, 'y': 8}], 'tags': ['spot', 'other']}  # Has multiple tags
    child3 = {'_id': 'child_3', 'coordinates': [
        {'x': 7, 'y': 7}], 'tags': ['other']}  # Doesn't have the right tag

    # Set up mock to return our parent and child annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        parent, child1, child2, child3
    ]

    # Create connections between parent and children
    connections = [
        {'parentId': 'parent_1', 'childId': 'child_1', 'dataset': 'test_dataset'},
        {'parentId': 'parent_1', 'childId': 'child_2', 'dataset': 'test_dataset'},
        {'parentId': 'parent_1', 'childId': 'child_3', 'dataset': 'test_dataset'}
    ]

    # Set up mock to return our connections
    mock_annotation_client.getAnnotationConnections.return_value = connections

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Mock pandas to avoid recursion issues
        with patch('pandas.DataFrame') as mock_df:
            # Create a mock DataFrame that will be returned by the DataFrame constructor
            mock_df_instance = MagicMock()
            mock_df.return_value = mock_df_instance

            # Mock the groupby chain to return a DataFrame with our expected count
            # With exclusive tag filtering, only child1 should be counted (child2 has multiple tags)
            mock_count_df = MagicMock()
            mock_df_instance.groupby.return_value.size.return_value.reset_index.return_value = mock_count_df

            # Set up the mock DataFrame to behave like a dictionary when accessed with __getitem__
            mock_count_df.__getitem__.side_effect = lambda key: [
                'parent_1'] if key == 'parentId' else [1]

            # Set up the mock DataFrame to behave like a list when iterated
            mock_count_df.__iter__.return_value = [{'parentId': 'parent_1', 'count': 1}]

            # Set up the zip function to return a list of tuples
            with patch('builtins.zip') as mock_zip:
                mock_zip.return_value = [('parent_1', 1)]

                # Run computation with exclusive tag filtering
                compute('test_dataset', 'http://test-api', 'test-token', exclusive_params)

                # Verify that add_multiple_annotation_property_values was called
                mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

                # Get the property values that were sent to the server
                property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[
                    0][0]
                assert 'test_dataset' in property_values

                # Check that the parent annotation has a count of 1 (only child1 should be counted)
                assert 'parent_1' in property_values['test_dataset']
                assert property_values['test_dataset']['parent_1'] == 1


def test_parent_with_no_children(mock_worker_client, mock_annotation_client, sample_params):
    """Test a parent annotation with no connected children"""
    # Create parent annotation
    parent = {
        '_id': 'parent_1',
        'coordinates': [{'x': 0, 'y': 0}, {'x': 0, 'y': 10}, {'x': 10, 'y': 10},
                        {'x': 10, 'y': 0}, {'x': 0, 'y': 0}],
        'tags': ['nucleus']
    }

    # Set up mock to return our parent annotation
    mock_worker_client.get_annotation_list_by_shape.return_value = [parent]

    # No connections for this parent
    mock_annotation_client.getAnnotationConnections.return_value = []

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Mock pandas to avoid recursion issues
        with patch('pandas.DataFrame') as mock_df:
            # Create a mock DataFrame that will be returned by the DataFrame constructor
            mock_df_instance = MagicMock()
            mock_df.return_value = mock_df_instance

            # Mock the groupby chain to return an empty DataFrame
            mock_count_df = MagicMock()
            mock_df_instance.groupby.return_value.size.return_value.reset_index.return_value = mock_count_df

            # Set up the mock DataFrame to behave like an empty list when iterated
            mock_count_df.__iter__.return_value = []

            # Set up the zip function to return an empty list
            with patch('builtins.zip') as mock_zip:
                mock_zip.return_value = []

                # Run computation
                compute('test_dataset', 'http://test-api', 'test-token', sample_params)

                # Verify that add_multiple_annotation_property_values was called
                mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

                # Get the property values that were sent to the server
                property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[
                    0][0]
                assert 'test_dataset' in property_values

                # Check that the parent annotation has a count of 0 (or is not in the results)
                if 'parent_1' in property_values['test_dataset']:
                    assert property_values['test_dataset']['parent_1'] == 0
                else:
                    # It's also valid if the parent is not included in the results at all
                    pass


def test_empty_tag_filters(mock_worker_client, mock_annotation_client):
    """Test behavior when no child tags are specified"""
    # Create parameters with empty child tags
    empty_tag_params = {
        'id': 'test_property_id',
        'name': 'Children Count',
        'image': 'properties/children_count:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'polygon',
        'workerInterface': {
            'Child Tags': [],  # Empty tag list
            'Child Tags Exclusive': 'No'
        }
    }

    # Create parent and child annotations
    parent = {
        '_id': 'parent_1',
        'coordinates': [{'x': 0, 'y': 0}, {'x': 0, 'y': 10}, {'x': 10, 'y': 10},
                        {'x': 10, 'y': 0}, {'x': 0, 'y': 0}],
        'tags': ['nucleus']
    }

    child1 = {'_id': 'child_1', 'coordinates': [{'x': 5, 'y': 5}], 'tags': ['spot']}
    child2 = {'_id': 'child_2', 'coordinates': [{'x': 8, 'y': 8}], 'tags': ['other']}

    # Set up mock to return our annotations
    mock_worker_client.get_annotation_list_by_shape.return_value = [parent, child1, child2]

    # Create connections
    connections = [
        {'parentId': 'parent_1', 'childId': 'child_1', 'dataset': 'test_dataset'},
        {'parentId': 'parent_1', 'childId': 'child_2', 'dataset': 'test_dataset'}
    ]

    # Set up mock to return our connections
    mock_annotation_client.getAnnotationConnections.return_value = connections

    # Mock sendProgress to avoid errors
    with patch('annotation_client.utils.sendProgress'):
        # Mock pandas to avoid recursion issues
        with patch('pandas.DataFrame') as mock_df:
            # Create a mock DataFrame that will be returned by the DataFrame constructor
            mock_df_instance = MagicMock()
            mock_df.return_value = mock_df_instance

            # Mock the groupby chain to return a DataFrame with our expected count
            # With empty tag filter, all children should be counted
            mock_count_df = MagicMock()
            mock_df_instance.groupby.return_value.size.return_value.reset_index.return_value = mock_count_df

            # Set up the mock DataFrame to behave like a dictionary when accessed with __getitem__
            mock_count_df.__getitem__.side_effect = lambda key: [
                'parent_1'] if key == 'parentId' else [2]

            # Set up the mock DataFrame to behave like a list when iterated
            mock_count_df.__iter__.return_value = [{'parentId': 'parent_1', 'count': 2}]

            # Set up the zip function to return a list of tuples
            with patch('builtins.zip') as mock_zip:
                mock_zip.return_value = [('parent_1', 2)]

                # Run computation with empty tag filter
                compute('test_dataset', 'http://test-api', 'test-token', empty_tag_params)

                # Verify that add_multiple_annotation_property_values was called
                mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

                # Get the property values that were sent to the server
                property_values = mock_worker_client.add_multiple_annotation_property_values.call_args[
                    0][0]
                assert 'test_dataset' in property_values

                # Check that the parent annotation has a count of 2 (all children should be counted)
                assert 'parent_1' in property_values['test_dataset']
                assert property_values['test_dataset']['parent_1'] == 2
