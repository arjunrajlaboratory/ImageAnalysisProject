import pytest
from unittest.mock import patch, MagicMock, call

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
        yield client


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_property_id',
        'name': 'test_point_count',
        'image': 'properties/blob_point_count:latest',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Tags of points to count': ['nucleus'],
            'Count points across all z-slices': 'Yes',
            'Exact tag match?': 'No'
        }
    }


@pytest.fixture
def sample_polygon():
    """Create a sample polygon annotation"""
    return {
        '_id': 'test_polygon',
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
        'tags': ['cell']
    }


@pytest.fixture
def sample_points():
    """Create sample point annotations"""
    # Points inside the polygon
    inside_points = [
        {
            '_id': f'point_inside_{i}',
            'coordinates': [{'x': i+1, 'y': i+1}],
            'location': {'Time': 0, 'Z': 0, 'XY': 0},
            'tags': ['nucleus']
        } for i in range(5)
    ]

    # Points outside the polygon
    outside_points = [
        {
            '_id': f'point_outside_{i}',
            'coordinates': [{'x': i+15, 'y': i+15}],
            'location': {'Time': 0, 'Z': 0, 'XY': 0},
            'tags': ['nucleus']
        } for i in range(3)
    ]

    # Points with different tags
    different_tag_points = [
        {
            '_id': f'point_different_tag_{i}',
            'coordinates': [{'x': i+2, 'y': i+2}],
            'location': {'Time': 0, 'Z': 0, 'XY': 0},
            'tags': ['cytoplasm']
        } for i in range(2)
    ]

    # Points in different z-slice
    different_z_points = [
        {
            '_id': f'point_different_z_{i}',
            'coordinates': [{'x': i+3, 'y': i+3}],
            'location': {'Time': 0, 'Z': 1, 'XY': 0},
            'tags': ['nucleus']
        } for i in range(3)
    ]

    return inside_points + outside_points + different_tag_points + different_z_points


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
        assert 'Tags of points to count' in interface_data
        assert interface_data['Tags of points to count']['type'] == 'tags'

        assert 'Count points across all z-slices' in interface_data
        assert interface_data['Count points across all z-slices']['type'] == 'select'
        assert 'Yes' in interface_data['Count points across all z-slices']['items']
        assert 'No' in interface_data['Count points across all z-slices']['items']
        assert interface_data['Count points across all z-slices']['default'] == 'Yes'

        assert 'Exact tag match?' in interface_data
        assert interface_data['Exact tag match?']['type'] == 'select'
        assert interface_data['Exact tag match?']['default'] == 'No'


def test_worker_startup(mock_worker_client, sample_params):
    """Test that the worker starts up correctly with no annotations"""
    # Run computation with empty annotation list
    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify that get_annotation_list_by_shape was called for both polygons and points
    assert mock_worker_client.get_annotation_list_by_shape.call_count == 2

    # First call should be for polygons
    assert mock_worker_client.get_annotation_list_by_shape.call_args_list[0][0][0] == 'polygon'

    # Second call should be for points
    assert mock_worker_client.get_annotation_list_by_shape.call_args_list[1][0][0] == 'point'

    # Since there are no annotations, add_multiple_annotation_property_values should not be called
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_basic_point_counting(mock_worker_client, sample_params, sample_polygon, sample_points):
    """Test basic point counting within a polygon"""
    # Set up mock to return our polygon and points
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_polygon],  # First call returns the polygon
        sample_points      # Second call returns all points
    ]

    # Mock the filter_elements_T_XY method to return all points
    with patch('annotation_utilities.annotation_tools.filter_elements_T_XY') as mock_filter:
        # Return only nucleus points
        nucleus_points = [p for p in sample_points if 'nucleus' in p['tags']]
        mock_filter.return_value = nucleus_points

        # Mock the create_points_from_annotations method
        with patch('annotation_utilities.annotation_tools.create_points_from_annotations') as mock_create_points:
            # Create mock shapely points
            mock_shapely_points = []
            for i in range(len(nucleus_points)):
                mock_point = MagicMock()
                mock_point.bounds = (i, i, i, i)  # Simple bounds for testing
                mock_shapely_points.append(mock_point)

            mock_create_points.return_value = mock_shapely_points

            # Mock the rtree index
            with patch('rtree.index.Index') as mock_index:
                mock_idx = mock_index.return_value

                # Mock the intersection method to return indices of points inside the polygon
                # For our test, we'll say the first 5 points are inside (0-4)
                mock_idx.intersection.return_value = range(5)

                # Mock the Polygon.contains method to return True for the first 5 points
                with patch('shapely.geometry.Polygon.contains') as mock_contains:
                    mock_contains.side_effect = lambda p: mock_shapely_points.index(p) < 5

                    # Run computation
                    compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Verify that add_multiple_annotation_property_values was called
    mock_worker_client.add_multiple_annotation_property_values.assert_called_once()

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values = calls[0][0][0]['test_dataset']

    # Verify that the polygon has a count
    assert 'test_polygon' in property_values

    # The count should be 5 (the number of points we mocked as inside)
    assert property_values['test_polygon'] == 5


def test_tag_filtering(mock_worker_client, sample_params, sample_polygon, sample_points):
    """Test filtering points by tags"""
    # Set up mock to return our polygon and points
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_polygon],  # First call returns the polygon
        sample_points      # Second call returns all points
    ]

    # Change the tag filter to 'cytoplasm'
    sample_params['workerInterface']['Tags of points to count'] = ['cytoplasm']

    # Mock the get_annotations_with_tags method to return only cytoplasm points
    with patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags:
        # First call returns all polygons (we don't filter those)
        # Second call returns only cytoplasm points
        cytoplasm_points = [p for p in sample_points if 'cytoplasm' in p['tags']]
        mock_get_tags.side_effect = [[sample_polygon], cytoplasm_points]

        # Mock the filter_elements_T_XY method to return all cytoplasm points
        with patch('annotation_utilities.annotation_tools.filter_elements_T_XY') as mock_filter:
            mock_filter.return_value = cytoplasm_points

            # Mock the create_points_from_annotations method
            with patch('annotation_utilities.annotation_tools.create_points_from_annotations') as mock_create_points:
                # Create mock shapely points
                mock_shapely_points = []
                for i in range(len(cytoplasm_points)):
                    mock_point = MagicMock()
                    mock_point.bounds = (i, i, i, i)  # Simple bounds for testing
                    mock_shapely_points.append(mock_point)

                mock_create_points.return_value = mock_shapely_points

                # Mock the rtree index
                with patch('rtree.index.Index') as mock_index:
                    mock_idx = mock_index.return_value

                    # Mock the intersection method to return indices of points inside the polygon
                    # For our test, we'll say both cytoplasm points are inside
                    mock_idx.intersection.return_value = range(len(cytoplasm_points))

                    # Mock the Polygon.contains method to return True for both points
                    with patch('shapely.geometry.Polygon.contains') as mock_contains:
                        mock_contains.return_value = True

                        # Run computation
                        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values that were sent to the server
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values = calls[0][0][0]['test_dataset']

    # The count should be 2 (the number of cytoplasm points)
    assert property_values['test_polygon'] == 2


def test_z_slice_filtering(mock_worker_client, sample_params, sample_polygon, sample_points):
    """Test counting points across z-slices vs. just the z-slice of the polygon"""
    # Set up mock to return our polygon and points
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_polygon],  # First call returns the polygon
        sample_points      # Second call returns all points
    ]

    # Test with counting across all z-slices (default)
    # Mock the get_annotations_with_tags method
    with patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags:
        # First call returns all polygons (we don't filter those)
        # Second call returns only nucleus points
        nucleus_points = [p for p in sample_points if 'nucleus' in p['tags']]
        mock_get_tags.side_effect = [[sample_polygon], nucleus_points]

        # Mock the filter_elements_T_XY method to return all nucleus points
        with patch('annotation_utilities.annotation_tools.filter_elements_T_XY') as mock_filter_txy:
            mock_filter_txy.return_value = nucleus_points

            # Mock the create_points_from_annotations method
            with patch('annotation_utilities.annotation_tools.create_points_from_annotations') as mock_create_points:
                # Create mock shapely points
                mock_shapely_points = []
                for i in range(len(nucleus_points)):
                    mock_point = MagicMock()
                    mock_point.bounds = (i, i, i, i)  # Simple bounds for testing
                    mock_shapely_points.append(mock_point)

                mock_create_points.return_value = mock_shapely_points

                # Mock the rtree index
                with patch('rtree.index.Index') as mock_index:
                    mock_idx = mock_index.return_value

                    # Mock the intersection method to return indices of all points
                    mock_idx.intersection.return_value = range(len(nucleus_points))

                    # Mock the Polygon.contains method to return True for all points
                    with patch('shapely.geometry.Polygon.contains') as mock_contains:
                        mock_contains.return_value = True

                        # Run computation
                        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values for counting across all z-slices
    calls_all_z = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values_all_z = calls_all_z[0][0][0]['test_dataset']

    # Reset the mock
    mock_worker_client.reset_mock()
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_polygon],  # First call returns the polygon
        sample_points      # Second call returns all points
    ]

    # Now test with counting only in the same z-slice
    sample_params['workerInterface']['Count points across all z-slices'] = 'No'

    # Mock the get_annotations_with_tags method
    with patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags:
        # First call returns all polygons (we don't filter those)
        # Second call returns only nucleus points
        nucleus_points = [p for p in sample_points if 'nucleus' in p['tags']]
        mock_get_tags.side_effect = [[sample_polygon], nucleus_points]

        # Mock the filter_elements_T_XY_Z method to return only nucleus points in z=0
        with patch('annotation_utilities.annotation_tools.filter_elements_T_XY_Z') as mock_filter_txyz:
            same_z_points = [p for p in nucleus_points if p['location']['Z'] == 0]
            mock_filter_txyz.return_value = same_z_points

            # Mock the create_points_from_annotations method
            with patch('annotation_utilities.annotation_tools.create_points_from_annotations') as mock_create_points:
                # Create mock shapely points
                mock_shapely_points = []
                for i in range(len(same_z_points)):
                    mock_point = MagicMock()
                    mock_point.bounds = (i, i, i, i)  # Simple bounds for testing
                    mock_shapely_points.append(mock_point)

                mock_create_points.return_value = mock_shapely_points

                # Mock the rtree index
                with patch('rtree.index.Index') as mock_index:
                    mock_idx = mock_index.return_value

                    # Mock the intersection method to return indices of all points
                    mock_idx.intersection.return_value = range(len(same_z_points))

                    # Mock the Polygon.contains method to return True for all points
                    with patch('shapely.geometry.Polygon.contains') as mock_contains:
                        mock_contains.return_value = True

                        # Run computation
                        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values for counting only in the same z-slice
    calls_same_z = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values_same_z = calls_same_z[0][0][0]['test_dataset']

    # Counting across all z-slices should include more points than counting only in the same z-slice
    # We have 11 nucleus points in total (8 in z=0, 3 in z=1)
    assert property_values_all_z['test_polygon'] == 11
    # We have 8 nucleus points in z=0 (5 inside + 3 outside)
    assert property_values_same_z['test_polygon'] == 8


def test_exact_tag_matching(mock_worker_client, sample_params, sample_polygon):
    """Test exact tag matching vs. partial tag matching"""
    # Create points with different tag combinations
    points_with_tags = [
        # Points with exact 'nucleus' tag
        {
            '_id': 'point_exact_1',
            'coordinates': [{'x': 2, 'y': 2}],
            'location': {'Time': 0, 'Z': 0, 'XY': 0},
            'tags': ['nucleus']
        },
        {
            '_id': 'point_exact_2',
            'coordinates': [{'x': 3, 'y': 3}],
            'location': {'Time': 0, 'Z': 0, 'XY': 0},
            'tags': ['nucleus']
        },
        # Points with 'nucleus' and other tags
        {
            '_id': 'point_multiple_1',
            'coordinates': [{'x': 4, 'y': 4}],
            'location': {'Time': 0, 'Z': 0, 'XY': 0},
            'tags': ['nucleus', 'bright']
        },
        {
            '_id': 'point_multiple_2',
            'coordinates': [{'x': 5, 'y': 5}],
            'location': {'Time': 0, 'Z': 0, 'XY': 0},
            'tags': ['nucleus', 'dim']
        },
        # Points with other tags
        {
            '_id': 'point_other',
            'coordinates': [{'x': 6, 'y': 6}],
            'location': {'Time': 0, 'Z': 0, 'XY': 0},
            'tags': ['cytoplasm']
        }
    ]

    # Set up mock to return our polygon and points
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_polygon],    # First call returns the polygon
        points_with_tags     # Second call returns all points
    ]

    # Test with non-exact tag matching (default)
    # Mock the get_annotations_with_tags method
    with patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags:
        # First call returns all polygons (we don't filter those)
        # Second call returns nucleus points (both exact and with other tags)
        non_exact_points = [p for p in points_with_tags if 'nucleus' in p['tags']]
        mock_get_tags.side_effect = [[sample_polygon], non_exact_points]

        # Mock the filter_elements_T_XY method
        with patch('annotation_utilities.annotation_tools.filter_elements_T_XY') as mock_filter:
            mock_filter.return_value = non_exact_points

            # Mock the create_points_from_annotations method
            with patch('annotation_utilities.annotation_tools.create_points_from_annotations') as mock_create_points:
                # Create mock shapely points
                mock_shapely_points = []
                for i in range(len(non_exact_points)):
                    mock_point = MagicMock()
                    mock_point.bounds = (i, i, i, i)  # Simple bounds for testing
                    mock_shapely_points.append(mock_point)

                mock_create_points.return_value = mock_shapely_points

                # Mock the rtree index
                with patch('rtree.index.Index') as mock_index:
                    mock_idx = mock_index.return_value

                    # Mock the intersection method to return indices of all points
                    mock_idx.intersection.return_value = range(len(non_exact_points))

                    # Mock the Polygon.contains method to return True for all points
                    with patch('shapely.geometry.Polygon.contains') as mock_contains:
                        mock_contains.return_value = True

                        # Run computation
                        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values for non-exact tag matching
    calls_non_exact = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values_non_exact = calls_non_exact[0][0][0]['test_dataset']

    # Reset the mock
    mock_worker_client.reset_mock()
    mock_worker_client.get_annotation_list_by_shape.side_effect = [
        [sample_polygon],    # First call returns the polygon
        points_with_tags     # Second call returns all points
    ]

    # Now test with exact tag matching
    sample_params['workerInterface']['Exact tag match?'] = 'Yes'

    # Mock the get_annotations_with_tags method
    with patch('annotation_utilities.annotation_tools.get_annotations_with_tags') as mock_get_tags:
        # First call returns all polygons (we don't filter those)
        # Second call returns only exact nucleus points
        exact_points = [p for p in points_with_tags if p['tags'] == ['nucleus']]
        mock_get_tags.side_effect = [[sample_polygon], exact_points]

        # Mock the filter_elements_T_XY method
        with patch('annotation_utilities.annotation_tools.filter_elements_T_XY') as mock_filter:
            mock_filter.return_value = exact_points

            # Mock the create_points_from_annotations method
            with patch('annotation_utilities.annotation_tools.create_points_from_annotations') as mock_create_points:
                # Create mock shapely points
                mock_shapely_points = []
                for i in range(len(exact_points)):
                    mock_point = MagicMock()
                    mock_point.bounds = (i, i, i, i)  # Simple bounds for testing
                    mock_shapely_points.append(mock_point)

                mock_create_points.return_value = mock_shapely_points

                # Mock the rtree index
                with patch('rtree.index.Index') as mock_index:
                    mock_idx = mock_index.return_value

                    # Mock the intersection method to return indices of all points
                    mock_idx.intersection.return_value = range(len(exact_points))

                    # Mock the Polygon.contains method to return True for all points
                    with patch('shapely.geometry.Polygon.contains') as mock_contains:
                        mock_contains.return_value = True

                        # Run computation
                        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

    # Get the property values for exact tag matching
    calls_exact = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values_exact = calls_exact[0][0][0]['test_dataset']

    # Non-exact matching should include points with 'nucleus' tag and other tags
    assert property_values_non_exact['test_polygon'] == 4  # 2 exact + 2 with multiple tags
    # Exact matching should only include points with exactly 'nucleus' tag
    assert property_values_exact['test_polygon'] == 2      # Only the 2 with exact tag
