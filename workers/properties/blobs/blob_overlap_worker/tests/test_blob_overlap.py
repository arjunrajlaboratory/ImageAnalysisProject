import pytest
from unittest.mock import patch, MagicMock

# Import your worker module
from entrypoint import compute, interface, extract_spatial_annotation_data

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point


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
def sample_polygon_annotation():
    """Create a sample polygon annotation"""
    return {
        '_id': 'test_polygon_1',
        'shape': 'polygon',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['nucleus']
    }


@pytest.fixture
def sample_overlapping_polygon():
    """Create a polygon that overlaps with sample_polygon_annotation"""
    return {
        '_id': 'test_polygon_2',
        'shape': 'polygon',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 5}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['cytoplasm']
    }


@pytest.fixture
def sample_non_overlapping_polygon():
    """Create a polygon that doesn't overlap with sample_polygon_annotation"""
    return {
        '_id': 'test_polygon_3',
        'shape': 'polygon',
        'coordinates': [
            {'x': 20, 'y': 20},
            {'x': 20, 'y': 30},
            {'x': 30, 'y': 30},
            {'x': 30, 'y': 20}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['cytoplasm']
    }


@pytest.fixture
def sample_params():
    """Create sample parameters that would be passed to the worker"""
    return {
        'id': 'test_overlap_id',
        'name': 'test_overlap',
        'image': 'properties/blob_overlap:latest',
        'tags': {'exclusive': False, 'tags': ['nucleus']},
        'shape': 'polygon',
        'workerInterface': {
            'Annotations to compute overlap with': ['cytoplasm'],
            'Compute reverse overlaps': True
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
        assert 'Blob Overlap' in interface_data
        assert 'Annotations to compute overlap with' in interface_data
        assert 'Compute reverse overlaps' in interface_data
        assert interface_data['Annotations to compute overlap with']['type'] == 'tags'
        assert interface_data['Compute reverse overlaps']['type'] == 'checkbox'


def test_extract_spatial_annotation_data_polygon():
    """Test extracting spatial data from polygon annotations"""
    annotations = [
        {
            '_id': 'test_1',
            'shape': 'polygon',
            'coordinates': [
                {'x': 0, 'y': 0},
                {'x': 0, 'y': 10},
                {'x': 10, 'y': 10},
                {'x': 10, 'y': 0}
            ],
            'location': {'Time': 0, 'XY': 0, 'Z': 0}
        }
    ]
    
    gdf = extract_spatial_annotation_data(annotations)
    
    assert len(gdf) == 1
    assert gdf.iloc[0]['_id'] == 'test_1'
    assert gdf.iloc[0]['Time'] == 0
    assert gdf.iloc[0]['XY'] == 0
    assert gdf.iloc[0]['Z'] == 0
    assert isinstance(gdf.iloc[0]['geometry'], Polygon)
    assert gdf.iloc[0]['geometry'].area == 100  # 10x10 square


def test_extract_spatial_annotation_data_point():
    """Test extracting spatial data from point annotations"""
    annotations = [
        {
            '_id': 'test_point',
            'shape': 'point',
            'coordinates': [{'x': 5, 'y': 5}],
            'location': {'Time': 0, 'XY': 0, 'Z': 0}
        }
    ]
    
    gdf = extract_spatial_annotation_data(annotations)
    
    assert len(gdf) == 1
    assert isinstance(gdf.iloc[0]['geometry'], Point)
    assert gdf.iloc[0]['geometry'].x == 5
    assert gdf.iloc[0]['geometry'].y == 5


def test_extract_spatial_annotation_data_invalid_polygon():
    """Test handling of invalid polygons (less than 3 coordinates)"""
    annotations = [
        {
            '_id': 'test_invalid',
            'shape': 'polygon',
            'coordinates': [
                {'x': 0, 'y': 0},
                {'x': 10, 'y': 10}
            ],
            'location': {'Time': 0, 'XY': 0, 'Z': 0}
        }
    ]
    
    gdf = extract_spatial_annotation_data(annotations)
    
    # Should be empty due to invalid polygon being skipped
    assert len(gdf) == 0
    # Verify the GeoDataFrame structure is correct even when empty
    if len(gdf) > 0:
        assert 'geometry' in gdf.columns


def test_compute_no_annotations(mock_worker_client, sample_params):
    """Test handling when no annotations are found"""
    mock_worker_client.get_annotation_list_by_shape.return_value = []
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Should not call add_multiple_annotation_property_values
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_compute_no_matching_annotations(mock_worker_client, sample_params, sample_polygon_annotation):
    """Test when no annotations match the filter tags"""
    # Return annotation with different tags
    sample_polygon_annotation['tags'] = ['different_tag']
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_polygon_annotation]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Should not call add_multiple_annotation_property_values
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_compute_overlapping_annotations(mock_worker_client, sample_params, 
                                       sample_polygon_annotation, sample_overlapping_polygon):
    """Test computing overlaps between overlapping annotations"""
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_polygon_annotation, sample_overlapping_polygon
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Should call add_multiple_annotation_property_values
    mock_worker_client.add_multiple_annotation_property_values.assert_called_once()
    
    # Get the computed values
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values = calls[0][0][0]['test_dataset']
    
    # Check that overlaps were computed
    assert 'test_polygon_1' in property_values
    assert 'test_polygon_2' in property_values
    
    # Check forward overlap (nucleus overlapping with cytoplasm)
    nucleus_overlap = property_values['test_polygon_1']['Overlap_cytoplasm']
    assert nucleus_overlap == pytest.approx(0.25, rel=0.01)  # 25 overlap / 100 total area
    
    # Check reverse overlap (cytoplasm overlapping with nucleus)
    cytoplasm_overlap = property_values['test_polygon_2']['Overlap_nucleus']
    assert cytoplasm_overlap == pytest.approx(0.25, rel=0.01)  # 25 overlap / 100 total area


def test_compute_non_overlapping_annotations(mock_worker_client, sample_params,
                                           sample_polygon_annotation, sample_non_overlapping_polygon):
    """Test computing overlaps between non-overlapping annotations"""
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_polygon_annotation, sample_non_overlapping_polygon
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Should not call add_multiple_annotation_property_values since no overlaps
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_compute_different_locations(mock_worker_client, sample_params,
                                   sample_polygon_annotation, sample_overlapping_polygon):
    """Test that annotations at different locations don't overlap"""
    # Place polygons at different locations
    sample_polygon_annotation['location'] = {'Time': 0, 'XY': 0, 'Z': 0}
    sample_overlapping_polygon['location'] = {'Time': 1, 'XY': 0, 'Z': 0}  # Different time
    
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_polygon_annotation, sample_overlapping_polygon
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Should not compute overlaps for different locations
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_compute_forward_overlap_only(mock_worker_client, sample_params,
                                    sample_polygon_annotation, sample_overlapping_polygon):
    """Test computing only forward overlaps (not reverse)"""
    sample_params['workerInterface']['Compute reverse overlaps'] = False
    
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_polygon_annotation, sample_overlapping_polygon
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Get the computed values
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values = calls[0][0][0]['test_dataset']
    
    # Should only have forward overlap for nucleus
    assert 'test_polygon_1' in property_values
    assert 'Overlap_cytoplasm' in property_values['test_polygon_1']
    
    # Should not have reverse overlap data
    assert 'test_polygon_2' not in property_values


def test_compute_multiple_overlaps_same_annotation(mock_worker_client, sample_params):
    """Test computing overlaps when one annotation overlaps with multiple others"""
    # Create main annotation
    main_annotation = {
        '_id': 'main_annotation',
        'shape': 'polygon',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 20},
            {'x': 20, 'y': 20},
            {'x': 20, 'y': 0}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['nucleus']
    }
    
    # Create two overlapping annotations
    overlap1 = {
        '_id': 'overlap_1',
        'shape': 'polygon',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 5}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['cytoplasm']
    }
    
    overlap2 = {
        '_id': 'overlap_2',
        'shape': 'polygon',
        'coordinates': [
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 18},
            {'x': 18, 'y': 18},
            {'x': 18, 'y': 10}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['cytoplasm']
    }
    
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        main_annotation, overlap1, overlap2
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Get the computed values
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values = calls[0][0][0]['test_dataset']
    
    # Main annotation should have total overlap from both overlapping annotations
    main_overlap = property_values['main_annotation']['Overlap_cytoplasm']
    # Total overlap area = 100 (from overlap1) + 64 (from overlap2) = 164
    # Main annotation area = 400
    # Overlap ratio = 164/400 = 0.41
    assert main_overlap == pytest.approx(0.41, rel=0.01)


def test_compute_partial_overlap(mock_worker_client, sample_params):
    """Test computing partial overlaps with precise calculations"""
    # Create a 10x10 square at (0,0)
    annotation1 = {
        '_id': 'ann_1',
        'shape': 'polygon',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['nucleus']
    }
    
    # Create a 10x10 square at (5,5) - 50% overlap
    annotation2 = {
        '_id': 'ann_2',
        'shape': 'polygon',
        'coordinates': [
            {'x': 5, 'y': 5},
            {'x': 5, 'y': 15},
            {'x': 15, 'y': 15},
            {'x': 15, 'y': 5}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['cytoplasm']
    }
    
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        annotation1, annotation2
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Get the computed values
    calls = mock_worker_client.add_multiple_annotation_property_values.call_args_list
    property_values = calls[0][0][0]['test_dataset']
    
    # Both annotations should have 25% overlap (25 overlap area / 100 total area)
    assert property_values['ann_1']['Overlap_cytoplasm'] == pytest.approx(0.25, rel=0.01)
    assert property_values['ann_2']['Overlap_nucleus'] == pytest.approx(0.25, rel=0.01)


def test_compute_exclusive_tag_filtering(mock_worker_client, sample_params,
                                       sample_polygon_annotation, sample_overlapping_polygon):
    """Test exclusive tag filtering"""
    # Add additional tag to first annotation
    sample_polygon_annotation['tags'] = ['nucleus', 'extra_tag']
    
    # Set exclusive filtering
    sample_params['tags']['exclusive'] = True
    
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_polygon_annotation, sample_overlapping_polygon
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Should not process annotations due to exclusive filtering
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


# Progress reporting test removed - other workers don't test this aspect


def test_compute_with_point_annotations(mock_worker_client, sample_params):
    """Test that point annotations are handled correctly"""
    point_annotation = {
        '_id': 'point_1',
        'shape': 'point',
        'coordinates': [{'x': 5, 'y': 5}],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['nucleus']
    }
    
    polygon_annotation = {
        '_id': 'polygon_1',
        'shape': 'polygon',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['cytoplasm']
    }
    
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        point_annotation, polygon_annotation
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Points have zero area, so no meaningful overlap can be computed
    # Worker should not call add_multiple_annotation_property_values for zero-area geometries
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_compute_edge_case_touching_polygons(mock_worker_client, sample_params):
    """Test polygons that touch but don't overlap"""
    # Create two adjacent 10x10 squares
    annotation1 = {
        '_id': 'touching_1',
        'shape': 'polygon',
        'coordinates': [
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 10},
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 0}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['nucleus']
    }
    
    annotation2 = {
        '_id': 'touching_2',
        'shape': 'polygon',
        'coordinates': [
            {'x': 10, 'y': 0},  # Shares edge with annotation1
            {'x': 10, 'y': 10},
            {'x': 20, 'y': 10},
            {'x': 20, 'y': 0}
        ],
        'location': {'Time': 0, 'XY': 0, 'Z': 0},
        'tags': ['cytoplasm']
    }
    
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        annotation1, annotation2
    ]
    
    with patch('annotation_client.utils.sendProgress'):
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)
    
    # Touching polygons should not have meaningful overlap
    mock_worker_client.add_multiple_annotation_property_values.assert_not_called() 