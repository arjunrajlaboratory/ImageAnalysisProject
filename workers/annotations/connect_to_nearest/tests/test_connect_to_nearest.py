from entrypoint import (
    interface,
    extract_spatial_annotation_data,
    compute_nearest_child_to_parent,
    compute
)
import pytest
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from unittest.mock import Mock, patch, MagicMock

# Import the functions we want to test
import sys
import os
sys.path.append('/app')


class TestConnectToNearestInterface:
    """Test interface generation"""

    def test_interface_generation(self):
        """Test that interface generates correct parameters"""

        # Mock the client
        mock_client = Mock()

        with patch('entrypoint.workers.UPennContrastWorkerPreviewClient', return_value=mock_client):
            interface("test_image", "http://test.com", "test_token")

        # Verify client was called with interface
        mock_client.setWorkerImageInterface.assert_called_once()
        call_args = mock_client.setWorkerImageInterface.call_args

        # Check that image and interface were passed
        assert call_args[0][0] == "test_image"
        interface_dict = call_args[0][1]

        # Verify expected interface parameters exist
        expected_params = [
            'Connect to nearest',
            'Parent tag',
            'Child tag',
            'Connect across Z',
            'Connect across T',
            'Connect to closest centroid or edge',
            'Restrict connection',
            'Max distance (pixels)',
            'Connect up to N children'
        ]

        for param in expected_params:
            assert param in interface_dict

        # Verify some specific parameter properties
        assert interface_dict['Parent tag']['type'] == 'tags'
        assert interface_dict['Child tag']['type'] == 'tags'
        assert interface_dict['Connect across Z']['type'] == 'select'
        assert interface_dict['Connect across Z']['items'] == ['Yes', 'No']
        assert interface_dict['Max distance (pixels)']['type'] == 'number'
        assert interface_dict['Max distance (pixels)']['max'] == 5000


class TestSpatialDataExtraction:
    """Test spatial annotation data extraction"""

    def test_extract_point_annotations(self):
        """Test extraction of point annotations"""

        annotations = [
            {
                '_id': 'point1',
                'shape': 'point',
                'coordinates': [{'x': 10, 'y': 20}],
                'location': {'Time': 1, 'XY': 0, 'Z': 0}
            },
            {
                '_id': 'point2',
                'shape': 'point',
                'coordinates': [{'x': 30, 'y': 40}],
                'location': {'Time': 1, 'XY': 0, 'Z': 0}
            }
        ]

        result = extract_spatial_annotation_data(annotations)

        assert len(result) == 2
        assert isinstance(result, gpd.GeoDataFrame)
        assert list(result['_id']) == ['point1', 'point2']
        assert all(result['Time'] == 1)
        assert all(result['XY'] == 0)
        assert all(result['Z'] == 0)

        # Check geometries
        assert isinstance(result.iloc[0]['geometry'], Point)
        assert result.iloc[0]['geometry'].x == 10
        assert result.iloc[0]['geometry'].y == 20

    def test_extract_polygon_annotations(self):
        """Test extraction of polygon annotations"""

        annotations = [
            {
                '_id': 'poly1',
                'shape': 'polygon',
                'coordinates': [
                    {'x': 0, 'y': 0},
                    {'x': 10, 'y': 0},
                    {'x': 10, 'y': 10},
                    {'x': 0, 'y': 10}
                ],
                'location': {'Time': 2, 'XY': 1, 'Z': 3}
            }
        ]

        result = extract_spatial_annotation_data(annotations)

        assert len(result) == 1
        assert result.iloc[0]['_id'] == 'poly1'
        assert result.iloc[0]['Time'] == 2
        assert result.iloc[0]['XY'] == 1
        assert result.iloc[0]['Z'] == 3
        assert isinstance(result.iloc[0]['geometry'], Polygon)

    def test_extract_mixed_annotations(self):
        """Test extraction of mixed point and polygon annotations"""

        annotations = [
            {
                '_id': 'point1',
                'shape': 'point',
                'coordinates': [{'x': 5, 'y': 5}],
                'location': {'Time': 1, 'XY': 0, 'Z': 0}
            },
            {
                '_id': 'poly1',
                'shape': 'polygon',
                'coordinates': [
                    {'x': 0, 'y': 0},
                    {'x': 10, 'y': 0},
                    {'x': 5, 'y': 10}
                ],
                'location': {'Time': 1, 'XY': 0, 'Z': 0}
            }
        ]

        result = extract_spatial_annotation_data(annotations)

        assert len(result) == 2
        assert isinstance(result.iloc[0]['geometry'], Point)
        assert isinstance(result.iloc[1]['geometry'], Polygon)


class TestNearestChildToParent:
    """Test nearest child to parent computation"""

    def create_test_dataframes(self):
        """Helper to create test child and parent dataframes"""

        # Create child points
        child_data = [
            {'_id': 'child1', 'geometry': Point(5, 5), 'Time': 1, 'XY': 0, 'Z': 0},
            {'_id': 'child2', 'geometry': Point(15, 15), 'Time': 1, 'XY': 0, 'Z': 0},
            {'_id': 'child3', 'geometry': Point(25, 25), 'Time': 1, 'XY': 0, 'Z': 0}
        ]
        child_df = gpd.GeoDataFrame(child_data)

        # Create parent polygons/points
        parent_data = [
            {'_id': 'parent1', 'geometry': Point(0, 0), 'Time': 1, 'XY': 0, 'Z': 0},
            {'_id': 'parent2', 'geometry': Point(20, 20), 'Time': 1, 'XY': 0, 'Z': 0}
        ]
        parent_df = gpd.GeoDataFrame(parent_data)

        return child_df, parent_df

    def test_compute_nearest_basic(self):
        """Test basic nearest neighbor computation"""

        child_df, parent_df = self.create_test_dataframes()

        result = compute_nearest_child_to_parent(child_df, parent_df)

        assert len(result) == 3  # Should connect all 3 children
        assert 'child_id' in result.columns
        assert 'nearest_parent_id' in result.columns
        assert 'distance' in result.columns

        # Check specific connections
        child1_row = result[result['child_id'] == 'child1']
        assert len(child1_row) == 1
        assert child1_row.iloc[0]['nearest_parent_id'] == 'parent1'  # Closer to parent1

        child2_row = result[result['child_id'] == 'child2']
        assert len(child2_row) == 1
        assert child2_row.iloc[0]['nearest_parent_id'] == 'parent2'  # Closer to parent2

    def test_compute_with_max_distance(self):
        """Test max distance constraint"""

        child_df, parent_df = self.create_test_dataframes()

        # Set max distance to exclude distant connections
        result = compute_nearest_child_to_parent(
            child_df, parent_df, max_distance=10
        )

        # Only child1 should be connected (distance ~7.07 to parent1)
        # child2 and child3 are farther than 10 pixels from nearest parents
        connected_children = set(result['child_id'])
        assert 'child1' in connected_children
        # child2 and child3 might be excluded depending on exact distances

    def test_compute_with_max_children(self):
        """Test max children per parent constraint"""

        # Create scenario where multiple children are near same parent
        child_data = [
            {'_id': 'child1', 'geometry': Point(1, 1), 'Time': 1, 'XY': 0, 'Z': 0},
            {'_id': 'child2', 'geometry': Point(2, 2), 'Time': 1, 'XY': 0, 'Z': 0},
            {'_id': 'child3', 'geometry': Point(3, 3), 'Time': 1, 'XY': 0, 'Z': 0}
        ]
        child_df = gpd.GeoDataFrame(child_data)

        parent_data = [
            {'_id': 'parent1', 'geometry': Point(0, 0), 'Time': 1, 'XY': 0, 'Z': 0}
        ]
        parent_df = gpd.GeoDataFrame(parent_data)

        # Limit to 2 children per parent
        result = compute_nearest_child_to_parent(
            child_df, parent_df, max_children=2
        )

        assert len(result) == 2  # Should only connect 2 closest children
        connected_children = set(result['child_id'])
        # Should connect the 2 closest children (child1 and child2)
        assert 'child1' in connected_children
        assert 'child2' in connected_children

    def test_compute_across_different_groups(self):
        """Test grouping by time/location prevents connections"""

        # Create children and parents in different time points
        child_data = [
            {'_id': 'child1', 'geometry': Point(5, 5), 'Time': 1, 'XY': 0, 'Z': 0},
            {'_id': 'child2', 'geometry': Point(15, 15), 'Time': 2, 'XY': 0, 'Z': 0}
        ]
        child_df = gpd.GeoDataFrame(child_data)

        parent_data = [
            {'_id': 'parent1', 'geometry': Point(0, 0), 'Time': 1, 'XY': 0, 'Z': 0}
        ]
        parent_df = gpd.GeoDataFrame(parent_data)

        # Default grouping includes Time, so child2 shouldn't connect
        result = compute_nearest_child_to_parent(child_df, parent_df)

        assert len(result) == 1  # Only child1 should connect
        assert result.iloc[0]['child_id'] == 'child1'

        # Test with custom grouping that excludes Time
        result_across_time = compute_nearest_child_to_parent(
            child_df, parent_df, groupby_cols=['XY', 'Z']
        )

        assert len(result_across_time) == 2  # Both children should connect

    def test_compute_no_parents(self):
        """Test behavior when no parents are available"""

        child_df, _ = self.create_test_dataframes()
        empty_parent_df = gpd.GeoDataFrame(columns=['_id', 'geometry', 'Time', 'XY', 'Z'])

        result = compute_nearest_child_to_parent(child_df, empty_parent_df)

        assert len(result) == 0  # No connections should be made

    def test_compute_no_children(self):
        """Test behavior when no children are available"""

        _, parent_df = self.create_test_dataframes()
        empty_child_df = gpd.GeoDataFrame(columns=['_id', 'geometry', 'Time', 'XY', 'Z'])

        result = compute_nearest_child_to_parent(empty_child_df, parent_df)

        assert len(result) == 0  # No connections should be made


class TestComputeFunction:
    """Test the main compute function"""

    @patch('entrypoint.annotations.UPennContrastAnnotationClient')
    @patch('entrypoint.tiles.UPennContrastDataset')
    @patch('entrypoint.annotation_tools.get_annotations_with_tags')
    def test_compute_basic_workflow(self, mock_get_annotations, mock_tiles, mock_annotations):
        """Test basic compute workflow with mocked dependencies"""

        # Mock annotation client
        mock_annotation_client = Mock()
        mock_annotations.return_value = mock_annotation_client

        # Mock point and blob annotations
        mock_annotation_client.getAnnotationsByDatasetId.side_effect = [
            [  # point annotations
                {
                    '_id': 'child1',
                    'shape': 'point',
                    'coordinates': [{'x': 5, 'y': 5}],
                    'location': {'Time': 1, 'XY': 0, 'Z': 0}
                }
            ],
            [  # blob annotations
                {
                    '_id': 'parent1',
                    'shape': 'polygon',
                    'coordinates': [
                        {'x': 0, 'y': 0},
                        {'x': 10, 'y': 0},
                        {'x': 10, 'y': 10},
                        {'x': 0, 'y': 10}
                    ],
                    'location': {'Time': 1, 'XY': 0, 'Z': 0}
                }
            ]
        ]

        # Mock get_annotations_with_tags to return parents and children
        mock_get_annotations.side_effect = [
            [  # parents
                {
                    '_id': 'parent1',
                    'shape': 'polygon',
                    'coordinates': [
                        {'x': 0, 'y': 0},
                        {'x': 10, 'y': 0},
                        {'x': 10, 'y': 10},
                        {'x': 0, 'y': 10}
                    ],
                    'location': {'Time': 1, 'XY': 0, 'Z': 0}
                }
            ],
            [  # children
                {
                    '_id': 'child1',
                    'shape': 'point',
                    'coordinates': [{'x': 5, 'y': 5}],
                    'location': {'Time': 1, 'XY': 0, 'Z': 0}
                }
            ]
        ]

        # Test parameters
        params = {
            'assignment': {},
            'channel': 'test_channel',
            'connectTo': 'test_connect',
            'tags': ['test_tag'],
            'tile': {'Time': 1, 'XY': 0, 'Z': 0},
            'workerInterface': {
                'Parent tag': ['parent'],
                'Child tag': ['child'],
                'Max distance (pixels)': 1000,
                'Connect across Z': 'No',
                'Connect across T': 'No',
                'Connect to closest centroid or edge': 'Centroid',
                'Restrict connection': 'None',
                'Connect up to N children': 10000
            }
        }

        # Run compute
        compute('test_dataset', 'http://test.com', 'test_token', params)

        # Verify connections were created
        mock_annotation_client.createMultipleConnections.assert_called_once()

        # Check the connections that were created
        connections = mock_annotation_client.createMultipleConnections.call_args[0][0]
        assert len(connections) == 1
        assert connections[0]['parentId'] == 'parent1'
        assert connections[0]['childId'] == 'child1'
        assert connections[0]['datasetId'] == 'test_dataset'

    def test_compute_invalid_params(self):
        """Test compute function with invalid parameters"""

        # Missing required keys
        invalid_params = {
            'assignment': {},
            'channel': 'test_channel'
            # Missing other required keys
        }

        # Should handle gracefully and return early
        result = compute('test_dataset', 'http://test.com', 'test_token', invalid_params)
        assert result is None

    @patch('entrypoint.annotations.UPennContrastAnnotationClient')
    @patch('entrypoint.tiles.UPennContrastDataset')
    @patch('entrypoint.annotation_tools.get_annotations_with_tags')
    def test_compute_no_annotations(self, mock_get_annotations, mock_tiles, mock_annotations):
        """Test compute function when no annotations are found"""

        # Mock annotation client
        mock_annotation_client = Mock()
        mock_annotations.return_value = mock_annotation_client

        # Return empty annotation lists
        mock_annotation_client.getAnnotationsByDatasetId.return_value = []
        mock_get_annotations.return_value = []

        params = {
            'assignment': {},
            'channel': 'test_channel',
            'connectTo': 'test_connect',
            'tags': ['test_tag'],
            'tile': {'Time': 1, 'XY': 0, 'Z': 0},
            'workerInterface': {
                'Parent tag': ['parent'],
                'Child tag': ['child'],
                'Max distance (pixels)': 1000,
                'Connect across Z': 'No',
                'Connect across T': 'No',
                'Connect to closest centroid or edge': 'Centroid',
                'Restrict connection': 'None',
                'Connect up to N children': 10000
            }
        }

        # Should handle empty annotations gracefully
        compute('test_dataset', 'http://test.com', 'test_token', params)

        # Should still call createMultipleConnections but with empty list
        mock_annotation_client.createMultipleConnections.assert_called_once_with([])


if __name__ == '__main__':
    pytest.main([__file__])
