import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from entrypoint import compute, interface


def test_interface():
    """Test the interface generation."""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]

        assert 'Random Squares' in interface_data
        assert 'Square size' in interface_data
        assert 'Number of squares' in interface_data
        assert 'Batch XY' in interface_data
        assert 'Batch Z' in interface_data
        assert 'Batch Time' in interface_data

        assert interface_data['Random Squares']['type'] == 'notes'
        assert interface_data['Square size']['type'] == 'number'
        assert interface_data['Number of squares']['type'] == 'number'
        assert interface_data['Batch XY']['type'] == 'text'
        assert interface_data['Batch Z']['type'] == 'text'
        assert interface_data['Batch Time']['type'] == 'text'


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_generates_squares(mock_annotation_client, mock_dataset_client):
    """Test that compute generates the correct number of square annotations."""
    mock_dataset_client.return_value.tiles = {
        'tileWidth': 1000,
        'tileHeight': 1000,
        'IndexRange': {'IndexXY': 1, 'IndexZ': 1, 'IndexT': 1, 'IndexC': 1},
    }
    mock_dataset_client.return_value.coordinatesToFrameIndex.return_value = 0
    mock_dataset_client.return_value.getRegion.return_value = MagicMock(
        squeeze=MagicMock(return_value=np.zeros((1000, 1000)))
    )

    mock_annotation_client.return_value.createMultipleAnnotations.return_value = [
        {'_id': f'id_{i}'} for i in range(10)
    ]

    params = {
        'assignment': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0,
        'connectTo': {'tags': []},
        'tags': ['test'],
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'workerInterface': {
            'Square size': 10,
            'Number of squares': 10,
        },
    }

    compute('test_dataset', 'http://test-api', 'test-token', params)

    mock_annotation_client.return_value.createMultipleAnnotations.assert_called_once()
    annotations = mock_annotation_client.return_value.createMultipleAnnotations.call_args[0][0]
    assert len(annotations) == 10
    for ann in annotations:
        assert ann['shape'] == 'polygon'
        assert len(ann['coordinates']) == 5  # 4 corners + closing point (shapely)


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_square_coordinates_within_bounds(mock_annotation_client, mock_dataset_client):
    """Test that generated squares stay within image bounds."""
    tile_width = 500
    tile_height = 500

    mock_dataset_client.return_value.tiles = {
        'tileWidth': tile_width,
        'tileHeight': tile_height,
        'IndexRange': {'IndexXY': 1, 'IndexZ': 1, 'IndexT': 1, 'IndexC': 1},
    }
    mock_dataset_client.return_value.coordinatesToFrameIndex.return_value = 0
    mock_dataset_client.return_value.getRegion.return_value = MagicMock(
        squeeze=MagicMock(return_value=np.zeros((500, 500)))
    )

    mock_annotation_client.return_value.createMultipleAnnotations.return_value = [
        {'_id': f'id_{i}'} for i in range(50)
    ]

    params = {
        'assignment': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0,
        'connectTo': {'tags': []},
        'tags': ['test'],
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'workerInterface': {
            'Square size': 20,
            'Number of squares': 50,
        },
    }

    compute('test_dataset', 'http://test-api', 'test-token', params)

    annotations = mock_annotation_client.return_value.createMultipleAnnotations.call_args[0][0]
    for ann in annotations:
        for coord in ann['coordinates']:
            assert 0 <= coord['x'] <= tile_width, f"x={coord['x']} out of bounds"
            assert 0 <= coord['y'] <= tile_height, f"y={coord['y']} out of bounds"
