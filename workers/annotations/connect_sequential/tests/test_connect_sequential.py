import pytest
from unittest.mock import patch, MagicMock

# Import the worker module
from entrypoint import compute, interface, extract_spatial_annotation_data, compute_nearest_child_to_parent, get_previous_objects


@pytest.fixture
def mock_annotation_client():
    """Mock the UPennContrastAnnotationClient"""
    with patch('annotation_client.annotations.UPennContrastAnnotationClient') as mock_client:
        client = mock_client.return_value
        # Set up default behaviors
        client.getAnnotationsByDatasetId.return_value = []
        client.createMultipleConnections.return_value = {}
        yield client


@pytest.fixture
def mock_worker_client():
    """Mock the UPennContrastWorkerPreviewClient for interface testing"""
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_client:
        client = mock_client.return_value
        yield client


@pytest.fixture
def sample_point_annotations():
    """Create sample point annotations across different time points"""
    return [
        {
            '_id': 'point_t0_1',
            'shape': 'point',
            'coordinates': [{'x': 10, 'y': 10}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'point_t1_1',
            'shape': 'point',
            'coordinates': [{'x': 12, 'y': 12}],
            'location': {'Time': 1, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'point_t2_1',
            'shape': 'point',
            'coordinates': [{'x': 15, 'y': 15}],
            'location': {'Time': 2, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        }
    ]


@pytest.fixture
def sample_z_annotations():
    """Create sample point annotations across different Z slices"""
    return [
        {
            '_id': 'point_z0_1',
            'shape': 'point',
            'coordinates': [{'x': 10, 'y': 10}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'point_z1_1',
            'shape': 'point',
            'coordinates': [{'x': 12, 'y': 12}],
            'location': {'Time': 0, 'XY': 0, 'Z': 1},
            'tags': ['cell']
        },
        {
            '_id': 'point_z2_1',
            'shape': 'point',
            'coordinates': [{'x': 15, 'y': 15}],
            'location': {'Time': 0, 'XY': 0, 'Z': 2},
            'tags': ['cell']
        }
    ]


@pytest.fixture
def sample_polygon_annotations():
    """Create sample polygon annotations across different time points"""
    return [
        {
            '_id': 'polygon_t0_1',
            'shape': 'polygon',
            'coordinates': [
                {'x': 0, 'y': 0},
                {'x': 0, 'y': 5},
                {'x': 5, 'y': 5},
                {'x': 5, 'y': 0},
                {'x': 0, 'y': 0}
            ],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['nucleus']
        },
        {
            '_id': 'polygon_t1_1',
            'shape': 'polygon',
            'coordinates': [
                {'x': 2, 'y': 2},
                {'x': 2, 'y': 7},
                {'x': 7, 'y': 7},
                {'x': 7, 'y': 2},
                {'x': 2, 'y': 2}
            ],
            'location': {'Time': 1, 'XY': 0, 'Z': 0},
            'tags': ['nucleus']
        }
    ]


@pytest.fixture
def sample_params_time():
    """Create sample parameters for connecting across Time"""
    return {
        'assignment': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 'DAPI',
        'connectTo': None,
        'tags': {'exclusive': False, 'tags': ['cell']},
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'workerInterface': {
            'Object to connect tag': ['cell'],
            'Connect sequentially across': 'Time',
            'Max distance (pixels)': 20
        }
    }


@pytest.fixture
def sample_params_z():
    """Create sample parameters for connecting across Z"""
    return {
        'assignment': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 'DAPI',
        'connectTo': None,
        'tags': {'exclusive': False, 'tags': ['cell']},
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'workerInterface': {
            'Object to connect tag': ['cell'],
            'Connect sequentially across': 'Z',
            'Max distance (pixels)': 20
        }
    }


def test_extract_spatial_annotation_data(sample_point_annotations, sample_polygon_annotations):
    """Test extraction of spatial data from annotations"""
    # Test with point annotations
    point_data = extract_spatial_annotation_data(sample_point_annotations)
    assert len(point_data) == 3
    assert point_data[0]['_id'] == 'point_t0_1'
    assert point_data[0]['x'] == 10
    assert point_data[0]['y'] == 10
    assert point_data[0]['Time'] == 0

    # Test with polygon annotations
    polygon_data = extract_spatial_annotation_data(sample_polygon_annotations)
    assert len(polygon_data) == 2
    assert polygon_data[0]['_id'] == 'polygon_t0_1'
    # Polygon should have centroid coordinates
    assert polygon_data[0]['x'] == 2.5  # centroid of square
    assert polygon_data[0]['y'] == 2.5


def test_interface(mock_worker_client):
    """Test the interface generation"""
    interface('test_image', 'http://test-api', 'test-token')

    # Verify interface was set
    mock_worker_client.setWorkerImageInterface.assert_called_once()

    # Verify interface structure
    interface_data = mock_worker_client.setWorkerImageInterface.call_args[0][1]
    assert 'Object to connect tag' in interface_data
    assert 'Connect sequentially across' in interface_data
    assert 'Max distance (pixels)' in interface_data
    assert interface_data['Object to connect tag']['type'] == 'tags'
    assert interface_data['Connect sequentially across']['type'] == 'select'
    assert interface_data['Max distance (pixels)']['type'] == 'number'
    assert interface_data['Connect sequentially across']['items'] == ['Time', 'Z']
    assert interface_data['Connect sequentially across']['default'] == 'Time'


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_basic_compute_time(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_point_annotations, sample_params_time):
    """Test basic compute functionality with sequential time connections"""
    # Set up mocks
    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        sample_point_annotations,  # point annotations
        []  # polygon annotations
    ]
    mock_get_annotations.return_value = sample_point_annotations

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_time)

    # Verify annotations were fetched
    assert mock_annotation_client.getAnnotationsByDatasetId.call_count == 2

    # Verify connections were created
    mock_annotation_client.createMultipleConnections.assert_called_once()
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]

    # Should have connections between consecutive time points
    assert len(connections) > 0
    assert all('parentId' in conn for conn in connections)
    assert all('childId' in conn for conn in connections)
    assert all('datasetId' in conn for conn in connections)
    assert all('tags' in conn for conn in connections)


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_basic_compute_z(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_z_annotations, sample_params_z):
    """Test basic compute functionality with sequential Z connections"""
    # Set up mocks
    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        sample_z_annotations,  # point annotations
        []  # polygon annotations
    ]
    mock_get_annotations.return_value = sample_z_annotations

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_z)

    # Verify annotations were fetched
    assert mock_annotation_client.getAnnotationsByDatasetId.call_count == 2

    # Verify connections were created
    mock_annotation_client.createMultipleConnections.assert_called_once()
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]

    # Should have connections between consecutive Z slices
    assert len(connections) > 0
    assert all('parentId' in conn for conn in connections)
    assert all('childId' in conn for conn in connections)


def test_compute_invalid_params(mock_annotation_client):
    """Test compute with invalid parameters"""
    invalid_params = {
        'assignment': {'XY': 0, 'Z': 0, 'Time': 0},
        # Missing required keys
    }

    # Should return early without processing
    compute('test_dataset', 'http://test-api', 'test-token', invalid_params)

    # Should not have called annotation client methods
    mock_annotation_client.getAnnotationsByDatasetId.assert_not_called()
    mock_annotation_client.createMultipleConnections.assert_not_called()


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_compute_no_annotations(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_params_time):
    """Test compute with no matching annotations"""
    # Set up mocks to return empty lists
    mock_annotation_client.getAnnotationsByDatasetId.return_value = []
    mock_get_annotations.return_value = []

    # Run compute
    compute('test_dataset', 'http://test-api', 'test-token', sample_params_time)

    # Should not create any connections when no annotations are found
    mock_annotation_client.createMultipleConnections.assert_called_once_with([])


def test_compute_nearest_child_to_parent():
    """Test the nearest child to parent computation function"""
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point

    # Create test data - children and parents in same spatial group (XY, Z) but different Time
    child_data = pd.DataFrame({
        '_id': ['child1', 'child2'],
        'x': [10, 20],
        'y': [10, 20],
        'Time': [1, 1],
        'XY': [0, 0],
        'Z': [0, 0]
    })
    child_gdf = gpd.GeoDataFrame(child_data, geometry=gpd.points_from_xy(child_data.x, child_data.y))

    parent_data = pd.DataFrame({
        '_id': ['parent1', 'parent2'],
        'x': [12, 18],
        'y': [12, 18],
        'Time': [0, 0],
        'XY': [0, 0],
        'Z': [0, 0]
    })
    parent_gdf = gpd.GeoDataFrame(parent_data, geometry=gpd.points_from_xy(parent_data.x, parent_data.y))

    # Test the function with groupby_cols that match the spatial grouping (XY, Z only)
    # This simulates how connect_sequential actually calls this function
    result = compute_nearest_child_to_parent(child_gdf, parent_gdf, groupby_cols=['XY', 'Z'])

    # Should have connections for both children
    assert len(result) == 2
    assert 'child_id' in result.columns
    assert 'nearest_parent_id' in result.columns


def test_get_previous_objects():
    """Test the get_previous_objects function"""
    import pandas as pd

    # Create test dataframe
    df = pd.DataFrame({
        '_id': ['obj1', 'obj2', 'obj3', 'obj4'],
        'Time': [0, 1, 2, 1],
        'Z': [0, 0, 0, 1],
        'XY': [0, 0, 0, 0]
    })

    # Test Time-based previous objects
    current_obj_time = {'Time': 2, 'Z': 0, 'XY': 0}
    previous_time = get_previous_objects(current_obj_time, df, 'Time')
    assert len(previous_time) == 2  # obj2 and obj4 both at Time=1
    assert all(previous_time['Time'] == 1)

    # Test Z-based previous objects
    current_obj_z = {'Time': 1, 'Z': 1, 'XY': 0}
    previous_z = get_previous_objects(current_obj_z, df, 'Z')
    assert len(previous_z) == 3  # obj1, obj2, obj3 all at Z=0
    assert all(previous_z['Z'] == 0)


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_distance_threshold_within_limit(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_params_time):
    """Test connections are made when objects are within distance threshold"""
    # Create annotations with objects close together
    close_annotations = [
        {
            '_id': 'close_t0_1',
            'shape': 'point',
            'coordinates': [{'x': 10, 'y': 10}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'close_t1_1',
            'shape': 'point',
            'coordinates': [{'x': 12, 'y': 12}],  # Distance ~2.83, within threshold of 20
            'location': {'Time': 1, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        }
    ]

    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        close_annotations,  # point annotations
        []  # polygon annotations
    ]
    mock_get_annotations.return_value = close_annotations

    compute('test_dataset', 'http://test-api', 'test-token', sample_params_time)

    # Should create connections
    mock_annotation_client.createMultipleConnections.assert_called_once()
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]
    assert len(connections) == 1


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_distance_threshold_beyond_limit(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_params_time):
    """Test no connections are made when objects exceed distance threshold"""
    # Create annotations with objects far apart
    far_annotations = [
        {
            '_id': 'far_t0_1',
            'shape': 'point',
            'coordinates': [{'x': 10, 'y': 10}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'far_t1_1',
            'shape': 'point',
            'coordinates': [{'x': 100, 'y': 100}],  # Distance ~127, beyond threshold of 20
            'location': {'Time': 1, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        }
    ]

    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        far_annotations,  # point annotations
        []  # polygon annotations
    ]
    mock_get_annotations.return_value = far_annotations

    compute('test_dataset', 'http://test-api', 'test-token', sample_params_time)

    # Should not create connections due to distance
    mock_annotation_client.createMultipleConnections.assert_called_once()
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]
    assert len(connections) == 0


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_multiple_spatial_groups(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_params_time):
    """Test connections across multiple spatial groups (different XY positions)"""
    # Create annotations in different XY positions
    multi_xy_annotations = [
        {
            '_id': 'xy0_t0_1',
            'shape': 'point',
            'coordinates': [{'x': 10, 'y': 10}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'xy0_t1_1',
            'shape': 'point',
            'coordinates': [{'x': 12, 'y': 12}],
            'location': {'Time': 1, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'xy1_t0_1',
            'shape': 'point',
            'coordinates': [{'x': 50, 'y': 50}],
            'location': {'Time': 0, 'XY': 1, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'xy1_t1_1',
            'shape': 'point',
            'coordinates': [{'x': 52, 'y': 52}],
            'location': {'Time': 1, 'XY': 1, 'Z': 0},
            'tags': ['cell']
        }
    ]

    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        multi_xy_annotations,  # point annotations
        []  # polygon annotations
    ]
    mock_get_annotations.return_value = multi_xy_annotations

    compute('test_dataset', 'http://test-api', 'test-token', sample_params_time)

    # Should create connections within each XY group
    mock_annotation_client.createMultipleConnections.assert_called_once()
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]
    assert len(connections) == 2  # One connection per XY group


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_mixed_annotation_types(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_params_time):
    """Test handling of mixed point and polygon annotations"""
    mixed_annotations = [
        {
            '_id': 'point_t0_1',
            'shape': 'point',
            'coordinates': [{'x': 10, 'y': 10}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'polygon_t1_1',
            'shape': 'polygon',
            'coordinates': [
                {'x': 8, 'y': 8},
                {'x': 8, 'y': 12},
                {'x': 12, 'y': 12},
                {'x': 12, 'y': 8},
                {'x': 8, 'y': 8}
            ],
            'location': {'Time': 1, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        }
    ]

    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        [mixed_annotations[0]],  # point annotations
        [mixed_annotations[1]]   # polygon annotations
    ]
    mock_get_annotations.return_value = mixed_annotations

    compute('test_dataset', 'http://test-api', 'test-token', sample_params_time)

    # Should handle mixed types and create connections
    mock_annotation_client.createMultipleConnections.assert_called_once()
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]
    assert len(connections) == 1


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_single_time_point(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_params_time):
    """Test behavior with annotations at only one time point"""
    single_time_annotations = [
        {
            '_id': 'single_t0_1',
            'shape': 'point',
            'coordinates': [{'x': 10, 'y': 10}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'single_t0_2',
            'shape': 'point',
            'coordinates': [{'x': 20, 'y': 20}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        }
    ]

    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        single_time_annotations,  # point annotations
        []  # polygon annotations
    ]
    mock_get_annotations.return_value = single_time_annotations

    compute('test_dataset', 'http://test-api', 'test-token', sample_params_time)

    # Should not create connections when all objects are at the same time point
    mock_annotation_client.createMultipleConnections.assert_called_once()
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]
    assert len(connections) == 0


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.utils.sendProgress')
@patch('annotation_utilities.annotation_tools.get_annotations_with_tags')
def test_no_previous_objects(mock_get_annotations, mock_send_progress, mock_dataset_client, mock_annotation_client, sample_params_time):
    """Test behavior when objects have no previous time point to connect to"""
    # All objects at Time=0, so no previous objects exist
    no_previous_annotations = [
        {
            '_id': 'first_t0_1',
            'shape': 'point',
            'coordinates': [{'x': 10, 'y': 10}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        },
        {
            '_id': 'first_t0_2',
            'shape': 'point',
            'coordinates': [{'x': 20, 'y': 20}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0},
            'tags': ['cell']
        }
    ]

    mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
        no_previous_annotations,  # point annotations
        []  # polygon annotations
    ]
    mock_get_annotations.return_value = no_previous_annotations

    compute('test_dataset', 'http://test-api', 'test-token', sample_params_time)

    # Should not create connections when no previous objects exist
    mock_annotation_client.createMultipleConnections.assert_called_once()
    connections = mock_annotation_client.createMultipleConnections.call_args[0][0]
    assert len(connections) == 0


def test_compute_nearest_child_to_parent_with_max_distance():
    """Test compute_nearest_child_to_parent with distance filtering"""
    import pandas as pd
    import geopandas as gpd

    # Create test data with one close and one far parent
    child_data = pd.DataFrame({
        '_id': ['child1'],
        'x': [10],
        'y': [10],
        'Time': [1],
        'XY': [0],
        'Z': [0]
    })
    child_gdf = gpd.GeoDataFrame(child_data, geometry=gpd.points_from_xy(child_data.x, child_data.y))

    parent_data = pd.DataFrame({
        '_id': ['close_parent', 'far_parent'],
        'x': [12, 100],  # close_parent distance ~2.83, far_parent distance ~127
        'y': [12, 100],
        'Time': [0, 0],
        'XY': [0, 0],
        'Z': [0, 0]
    })
    parent_gdf = gpd.GeoDataFrame(parent_data, geometry=gpd.points_from_xy(parent_data.x, parent_data.y))

    # Test with max_distance that excludes far parent - use spatial grouping only
    result = compute_nearest_child_to_parent(child_gdf, parent_gdf, groupby_cols=['XY', 'Z'], max_distance=20)

    # Should only connect to close parent
    assert len(result) == 1
    assert result.iloc[0]['nearest_parent_id'] == 'close_parent'

    # Test with max_distance that excludes all parents
    result_none = compute_nearest_child_to_parent(child_gdf, parent_gdf, groupby_cols=['XY', 'Z'], max_distance=1)

    # Should have no connections
    assert len(result_none) == 0 