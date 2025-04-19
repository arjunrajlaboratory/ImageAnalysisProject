import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon

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
        client.add_multiple_annotation_property_values.return_value = None
        yield client


@pytest.fixture
def mock_dataset_client():
    """Mock the UPennContrastDataset"""
    with patch(
        'annotation_client.tiles.UPennContrastDataset'
    ) as mock_client:
        client = mock_client.return_value
        # Set up default behaviors
        client.getRegion.return_value = np.ones((50, 50), dtype=np.uint8) * 100
        client.coordinatesToFrameIndex.return_value = 0
        client.tiles = {'IndexRange': {'IndexZ': 1, 'IndexT': 1, 'IndexXY': 1, 'IndexC': 1}}
        yield client


@pytest.fixture
def sample_annotation_labeled():
    """Create a sample labeled polygon annotation (e.g., a square)"""
    return {
        '_id': 'test_id_labeled',
        'coordinates': [
            {'x': 10, 'y': 10},
            {'x': 10, 'y': 20},
            {'x': 20, 'y': 20},
            {'x': 20, 'y': 10},
            {'x': 10, 'y': 10}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell', 'classA']
    }


@pytest.fixture
def sample_annotation_unlabeled():
    """Create a sample unlabeled polygon annotation (e.g., a rectangle)"""
    return {
        '_id': 'test_id_unlabeled',
        'coordinates': [
            {'x': 30, 'y': 30},
            {'x': 30, 'y': 45},
            {'x': 40, 'y': 45},
            {'x': 40, 'y': 30},
            {'x': 30, 'y': 30}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell']
    }


@pytest.fixture
def sample_params():
    """Create sample parameters for the random forest classifier worker"""
    return {
        'id': 'test_property_id',
        'name': 'test_classification',
        'tags': {'exclusive': False, 'tags': ['cell']},
        'shape': 'polygon',
        'workerInterface': {
            'Buffer radius': 10,
            'Include texture features': True,
            'Texture scaling': 8
        }
    }


@pytest.fixture
def another_labeled_annotation():
    """Create a second sample labeled polygon annotation."""
    return {
        '_id': 'test_id_labeled_2',
        'coordinates': [
            {'x': 5, 'y': 5}, {'x': 5, 'y': 15}, {'x': 15, 'y': 15},
            {'x': 15, 'y': 5}, {'x': 5, 'y': 5}
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell', 'classB']  # Different class
    }


def test_interface():
    """Test the interface generation for the random forest classifier"""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]
        assert 'Buffer radius' in interface_data
        assert interface_data['Buffer radius']['type'] == 'number'
        assert interface_data['Buffer radius']['default'] == 10
        assert 'Include texture features' in interface_data
        assert interface_data['Include texture features']['type'] == 'checkbox'
        assert interface_data['Include texture features']['default'] is True
        assert 'Texture scaling' in interface_data
        assert interface_data['Texture scaling']['type'] == 'number'
        assert interface_data['Texture scaling']['default'] == 8


def test_worker_startup(mock_worker_client, mock_dataset_client, sample_params):
    empty_params = sample_params.copy()
    empty_params['tags'] = {'exclusive': True, 'tags': ['nonexistent']}
    with patch('entrypoint.sendWarning') as mock_send_warning:
        compute('test_dataset', 'http://test-api', 'test-token', empty_params)

        mock_worker_client.get_annotation_list_by_shape.assert_called_once_with(
            'polygon', limit=0)

        mock_send_warning.assert_called_once_with(
            'No objects found', info='No objects found. Please check the tags and shape.')

        mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_no_labeled_data(mock_worker_client, mock_dataset_client, sample_params, sample_annotation_unlabeled):
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_annotation_unlabeled
    ]

    with patch('entrypoint.sendWarning') as mock_send_warning:
        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        mock_send_warning.assert_called_once_with(
            'No labeled data', info='No labeled data found. Please tag some annotations.'
        )
        mock_worker_client.add_multiple_annotation_property_values.assert_not_called()


def test_invalid_polygon(mock_worker_client, mock_dataset_client, sample_params, sample_annotation_labeled, another_labeled_annotation, capsys):
    """Test skipping invalid polygons AND error handling for insufficient samples per class."""
    invalid_annotation = {
        '_id': 'invalid_polygon',
        'coordinates': [{'x': 0, 'y': 0}, {'x': 10, 'y': 10}],  # Only 2 points
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['cell', 'classA']  # Tagged to potentially be included
    }

    # Return valid labeled annotations (1 sample per class) and one invalid one
    # This setup should first filter out the invalid polygon, then trigger the
    # ValueError in train_test_split because each remaining class has only 1 sample.
    mock_worker_client.get_annotation_list_by_shape.return_value = [
        sample_annotation_labeled,    # classA
        another_labeled_annotation,  # classB
        invalid_annotation
    ]

    # Patch both sendWarning (for the invalid polygon) and sendError (for the classification error)
    with patch('entrypoint.sendWarning') as mock_send_warning, \
            patch('entrypoint.sendError') as mock_send_error:

        compute('test_dataset', 'http://test-api', 'test-token', sample_params)

        # Verify warning was sent for the invalid polygon
        mock_send_warning.assert_any_call(
            'Invalid polygon', info='Object invalid_polygon has less than 3 vertices.'
        )

        # Verify error was sent due to insufficient samples per class
        expected_error_message = (
            "Classification failed: At least one class has fewer than 2 labeled samples. "
            "Please ensure each class you want to predict has at least 2 labeled examples."
        )
        mock_send_error.assert_called_once_with(expected_error_message)

        # Verify properties were NOT added because the process stopped after the error
        mock_worker_client.add_multiple_annotation_property_values.assert_not_called()

# TODO: Add tests for 'Include texture features' toggle if needed
# TODO: Add tests for tag filtering ('exclusive' flag) if needed
