import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from entrypoint import compute, interface


# All interface types that should be demonstrated
EXPECTED_INTERFACE_TYPES = {
    'About this worker': 'notes',
    'Sample number': 'number',
    'Sample text': 'text',
    'Sample select': 'select',
    'Sample checkbox': 'checkbox',
    'Sample channel': 'channel',
    'Sample channel checkboxes': 'channelCheckboxes',
    'Sample tags': 'tags',
    'Sample layer': 'layer',
    'Batch XY': 'text',
    'Batch Z': 'text',
    'Batch Time': 'text',
}


def test_interface():
    """Test the interface generation contains all expected fields."""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        mock_client.return_value.setWorkerImageInterface.assert_called_once()

        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]

        for field_name, field_type in EXPECTED_INTERFACE_TYPES.items():
            assert field_name in interface_data, f"Missing field: {field_name}"
            assert interface_data[field_name]['type'] == field_type, \
                f"Field {field_name} has type {interface_data[field_name]['type']}, expected {field_type}"


def test_interface_display_order():
    """Test that all interface fields have a displayOrder set."""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]

        for field_name, field_config in interface_data.items():
            assert 'displayOrder' in field_config, \
                f"Field {field_name} is missing displayOrder"


def test_interface_tooltips():
    """Test that most interface fields have tooltips."""
    with patch(
        'annotation_client.workers.UPennContrastWorkerPreviewClient'
    ) as mock_client:
        interface('test_image', 'http://test-api', 'test-token')

        interface_data = mock_client.return_value.setWorkerImageInterface.call_args[0][1]

        # All fields except notes and batch fields should have tooltips
        for field_name, field_config in interface_data.items():
            if field_config['type'] not in ('notes', 'text'):
                assert 'tooltip' in field_config, \
                    f"Field {field_name} is missing tooltip"


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_sends_messages(mock_annotation_client, mock_dataset_client, capsys):
    """Test that compute sends progress, warning, and error messages."""
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
        {'_id': f'id_{i}'} for i in range(5)
    ]

    params = {
        'assignment': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0,
        'connectTo': {'tags': []},
        'tags': ['test'],
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'workerInterface': {
            'Sample number': 10,
            'Sample text': 'test',
            'Sample select': 'Option A',
            'Sample checkbox': False,
        },
    }

    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()

    # Check that progress, warning, and error messages were sent
    assert '"progress"' in captured.out
    assert '"warning"' in captured.out
    assert '"error"' in captured.out


@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_checkbox_sends_extra_warning(mock_annotation_client, mock_dataset_client, capsys):
    """Test that checking the checkbox triggers an extra warning."""
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
        {'_id': f'id_{i}'} for i in range(5)
    ]

    params = {
        'assignment': {'XY': 0, 'Z': 0, 'Time': 0},
        'channel': 0,
        'connectTo': {'tags': []},
        'tags': ['test'],
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'workerInterface': {
            'Sample number': 10,
            'Sample text': 'test',
            'Sample select': 'Option A',
            'Sample checkbox': True,
        },
    }

    compute('test_dataset', 'http://test-api', 'test-token', params)

    captured = capsys.readouterr()
    assert 'Checkbox was checked' in captured.out
