import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from entrypoint import compute, interface


@pytest.fixture
def mock_worker_client():
    with patch('annotation_client.workers.UPennContrastWorkerClient') as mock_cls:
        client = mock_cls.return_value
        client.get_annotation_list_by_shape.return_value = []
        yield client


@pytest.fixture
def mock_annotation_client():
    with patch('annotation_client.annotations.UPennContrastAnnotationClient') as mock_cls:
        client = mock_cls.return_value
        client.client = MagicMock()
        client.client.getFolder.return_value = {'_id': 'folder_id'}
        yield client


@pytest.fixture
def mock_dataset_client():
    with patch('annotation_client.tiles.UPennContrastDataset') as mock_cls:
        client = mock_cls.return_value
        client.tiles = {}
        client.coordinatesToFrameIndex.side_effect = (
            lambda xy, z, time, channel: channel
        )
        yield client


@pytest.fixture
def sample_line_annotation():
    return {
        '_id': 'line_1',
        'shape': 'line',
        'coordinates': [
            {'x': 1.5, 'y': 1.5},
            {'x': 5.5, 'y': 1.5},
        ],
        'location': {'Time': 0, 'Z': 0, 'XY': 0},
        'tags': ['scan'],
    }


def _params(all_channels, selected_channel=0, file_name='out.csv'):
    return {
        'id': 'prop_id',
        'name': 'line_scan_test',
        'image': 'properties/line_scan_worker:latest',
        'tags': {'exclusive': False, 'tags': []},
        'shape': 'line',
        'workerInterface': {
            'All channels': all_channels,
            'Channel': selected_channel,
            'File name': file_name,
        },
    }


def test_interface():
    """Interface registers required fields."""
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_cls:
        interface('img', 'http://api', 'token')
        mock_cls.return_value.setWorkerImageInterface.assert_called_once()
        interface_data = mock_cls.return_value.setWorkerImageInterface.call_args[0][1]
        for key in ('All channels', 'Channel', 'File name', 'Line Scan CSV'):
            assert key in interface_data


def test_no_annotations_returns_early(
    mock_worker_client, mock_annotation_client, mock_dataset_client
):
    """When no line annotations exist, no upload happens."""
    mock_worker_client.get_annotation_list_by_shape.return_value = []

    compute('dataset_id', 'http://api', 'token', _params(all_channels=True))

    mock_annotation_client.client.uploadStreamToFolder.assert_not_called()


def test_single_channel_dataset_no_IndexRange(
    mock_worker_client, mock_annotation_client, mock_dataset_client,
    sample_line_annotation,
):
    """Regression: single-frame datasets omit IndexRange entirely.

    Before the fix this raised KeyError: 'IndexRange'. After the fix it
    treats the dataset as having a single channel.
    """
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_line_annotation]
    mock_dataset_client.tiles = {}  # no IndexRange — the bug condition
    mock_dataset_client.getRegion.return_value = np.full((10, 10), 42, dtype=np.uint8)

    compute('dataset_id', 'http://api', 'token', _params(all_channels=True))

    # One channel was loaded.
    assert mock_dataset_client.getRegion.call_count == 1
    # Frame index resolved for channel 0.
    mock_dataset_client.coordinatesToFrameIndex.assert_called_once_with(0, 0, 0, 0)
    # CSV was uploaded.
    mock_annotation_client.client.uploadStreamToFolder.assert_called_once()


def test_multi_channel_dataset_with_IndexRange(
    mock_worker_client, mock_annotation_client, mock_dataset_client,
    sample_line_annotation,
):
    """When IndexRange.IndexC is set, all channels are scanned."""
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_line_annotation]
    mock_dataset_client.tiles = {'IndexRange': {'IndexC': 3}}
    mock_dataset_client.getRegion.return_value = np.full((10, 10), 7, dtype=np.uint8)

    compute('dataset_id', 'http://api', 'token', _params(all_channels=True))

    # Three channels were loaded.
    assert mock_dataset_client.getRegion.call_count == 3
    channels_loaded = [
        call.args[3] for call in mock_dataset_client.coordinatesToFrameIndex.call_args_list
    ]
    assert channels_loaded == [0, 1, 2]


def test_IndexRange_without_IndexC(
    mock_worker_client, mock_annotation_client, mock_dataset_client,
    sample_line_annotation,
):
    """IndexRange present but missing IndexC also defaults to 1 channel."""
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_line_annotation]
    mock_dataset_client.tiles = {'IndexRange': {'IndexZ': 4}}  # has IndexZ but no IndexC
    mock_dataset_client.getRegion.return_value = np.full((10, 10), 1, dtype=np.uint8)

    compute('dataset_id', 'http://api', 'token', _params(all_channels=True))

    assert mock_dataset_client.getRegion.call_count == 1


def test_all_channels_false_uses_selected_channel(
    mock_worker_client, mock_annotation_client, mock_dataset_client,
    sample_line_annotation,
):
    """When 'All channels' is unchecked, only the selected channel is scanned."""
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_line_annotation]
    mock_dataset_client.tiles = {'IndexRange': {'IndexC': 5}}
    mock_dataset_client.getRegion.return_value = np.full((10, 10), 99, dtype=np.uint8)

    compute(
        'dataset_id', 'http://api', 'token',
        _params(all_channels=False, selected_channel=2),
    )

    assert mock_dataset_client.getRegion.call_count == 1
    mock_dataset_client.coordinatesToFrameIndex.assert_called_once_with(0, 0, 0, 2)


def test_intensity_profile_values(
    mock_worker_client, mock_annotation_client, mock_dataset_client,
    sample_line_annotation,
):
    """Intensity profile along a constant-valued image is constant."""
    mock_worker_client.get_annotation_list_by_shape.return_value = [sample_line_annotation]
    mock_dataset_client.tiles = {}
    constant_image = np.full((10, 10), 123, dtype=np.uint8)
    mock_dataset_client.getRegion.return_value = constant_image

    compute('dataset_id', 'http://api', 'token', _params(all_channels=True))

    # Inspect the uploaded CSV stream.
    upload_call = mock_annotation_client.client.uploadStreamToFolder.call_args
    assert upload_call is not None
    csv_stream = upload_call.args[1]
    csv_stream.seek(0)
    csv_text = csv_stream.read()
    # Channel 0 row should contain repeated 123s.
    assert 'Channel 0' in csv_text
    assert '123' in csv_text
