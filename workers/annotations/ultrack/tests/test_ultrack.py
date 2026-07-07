import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from entrypoint import (
    interface,
    compute,
    group_annotations_by_location,
    build_label_stack,
    tracks_df_to_connections,
)


@pytest.fixture
def mock_worker_preview_client():
    with patch('annotation_client.workers.UPennContrastWorkerPreviewClient') as mock_client:
        yield mock_client.return_value


@pytest.fixture
def square_annotations():
    """One 4x4 square per time point, drifting diagonally over three frames."""
    def square(_id, t, x0, y0):
        return {
            '_id': _id,
            'shape': 'polygon',
            'coordinates': [
                {'x': x0, 'y': y0},
                {'x': x0 + 4, 'y': y0},
                {'x': x0 + 4, 'y': y0 + 4},
                {'x': x0, 'y': y0 + 4},
                {'x': x0, 'y': y0},
            ],
            'location': {'Time': t, 'XY': 0, 'Z': 0},
            'tags': ['cell'],
        }
    return [
        square('t0', 0, 2, 2),
        square('t1', 1, 4, 4),
        square('t2', 2, 6, 6),
    ]


def test_interface(mock_worker_preview_client):
    interface('test_image', 'http://api', 'token')
    mock_worker_preview_client.setWorkerImageInterface.assert_called_once()
    interface_data = mock_worker_preview_client.setWorkerImageInterface.call_args[0][1]
    assert 'Tag of objects to track' in interface_data
    assert 'Max distance' in interface_data
    assert 'Allow divisions' in interface_data
    assert interface_data['Tag of objects to track']['type'] == 'tags'
    assert interface_data['Allow divisions']['type'] == 'checkbox'


def test_group_annotations_by_location():
    anns = [
        {'shape': 'polygon', 'coordinates': [1, 2, 3], 'location': {'XY': 0, 'Z': 0, 'Time': 0}},
        {'shape': 'polygon', 'coordinates': [1, 2, 3], 'location': {'XY': 1, 'Z': 2, 'Time': 0}},
    ]
    groups = group_annotations_by_location(anns)
    assert set(groups.keys()) == {(0, 0), (1, 2)}


def test_build_label_stack(square_annotations):
    masks, label_to_id, label_centroid = build_label_stack(
        square_annotations, [0, 1, 2], height=20, width=20)
    assert masks.shape == (3, 20, 20)
    assert label_to_id[(0, 1)] == 't0'
    assert label_to_id[(2, 1)] == 't2'
    row, col = label_centroid[(1, 1)]
    # Second square spans x 4..8, y 4..8 -> centroid near (5.5, 5.5).
    assert abs(row - 5.5) < 1.5
    assert abs(col - 5.5) < 1.5


def _centroid_maps():
    """Simple centroid maps: one object per frame at increasing positions."""
    label_centroid = {
        (0, 1): (4.0, 4.0),
        (1, 1): (6.0, 6.0),
        (2, 1): (8.0, 8.0),
    }
    label_to_id = {(0, 1): 't0', (1, 1): 't1', (2, 1): 't2'}
    return label_centroid, label_to_id


def test_tracks_df_to_connections_linear():
    label_centroid, label_to_id = _centroid_maps()
    tracks_df = pd.DataFrame([
        {'track_id': 1, 't': 0, 'y': 4.0, 'x': 4.0, 'parent_track_id': -1},
        {'track_id': 1, 't': 1, 'y': 6.0, 'x': 6.0, 'parent_track_id': -1},
        {'track_id': 1, 't': 2, 'y': 8.0, 'x': 8.0, 'parent_track_id': -1},
    ])
    conns = tracks_df_to_connections(tracks_df, label_centroid, label_to_id, 'ds', ['Ultrack'])
    pairs = {(c['parentId'], c['childId']) for c in conns}
    assert pairs == {('t0', 't1'), ('t1', 't2')}


def test_tracks_df_to_connections_division():
    """Track 1 (t0,t1) divides into tracks 2 and 3 at t2."""
    label_centroid = {
        (0, 1): (4.0, 4.0),
        (1, 1): (6.0, 6.0),
        (2, 1): (8.0, 8.0),   # daughter A
        (2, 2): (2.0, 2.0),   # daughter B
    }
    label_to_id = {(0, 1): 'p0', (1, 1): 'p1', (2, 1): 'dA', (2, 2): 'dB'}
    tracks_df = pd.DataFrame([
        {'track_id': 1, 't': 0, 'y': 4.0, 'x': 4.0, 'parent_track_id': -1},
        {'track_id': 1, 't': 1, 'y': 6.0, 'x': 6.0, 'parent_track_id': -1},
        {'track_id': 2, 't': 2, 'y': 8.0, 'x': 8.0, 'parent_track_id': 1},
        {'track_id': 3, 't': 2, 'y': 2.0, 'x': 2.0, 'parent_track_id': 1},
    ])
    conns = tracks_df_to_connections(tracks_df, label_centroid, label_to_id, 'ds', [])
    pairs = {(c['parentId'], c['childId']) for c in conns}
    # Motion link p0->p1, plus division links p1->dA and p1->dB.
    assert pairs == {('p0', 'p1'), ('p1', 'dA'), ('p1', 'dB')}


def test_tracks_df_zero_parent_id_is_not_a_real_parent():
    """A founder track with parent_track_id == 0 must not be linked to track 0."""
    label_centroid = {
        (0, 1): (4.0, 4.0),   # track 0 node at t0
        (1, 1): (6.0, 6.0),   # track 0 node at t1
        (0, 2): (50.0, 50.0),  # founder track 5 node at t0
    }
    label_to_id = {(0, 1): 'z0', (1, 1): 'z1', (0, 2): 'founder'}
    # Track 0 is a real track; track 5 is a founder whose parent_track_id is 0.
    tracks_df = pd.DataFrame([
        {'track_id': 0, 't': 0, 'y': 4.0, 'x': 4.0, 'parent_track_id': -1},
        {'track_id': 0, 't': 1, 'y': 6.0, 'x': 6.0, 'parent_track_id': -1},
        {'track_id': 5, 't': 0, 'y': 50.0, 'x': 50.0, 'parent_track_id': 0},
    ])
    conns = tracks_df_to_connections(tracks_df, label_centroid, label_to_id, 'ds', [])
    pairs = {(c['parentId'], c['childId']) for c in conns}
    # Only the motion link within track 0; the founder must NOT be wired to track 0.
    assert pairs == {('z0', 'z1')}
    assert ('z1', 'founder') not in pairs


def test_tracks_df_nearest_match_assigns_correct_annotation():
    """Each row is mapped to the annotation whose mask centroid is nearest."""
    # Two objects per frame: 'A' near the origin, 'B' near (100,100).
    label_centroid = {
        (0, 1): (0.0, 0.0), (0, 2): (100.0, 100.0),
        (1, 1): (0.0, 0.0), (1, 2): (100.0, 100.0),
    }
    label_to_id = {
        (0, 1): 'A0', (0, 2): 'B0',
        (1, 1): 'A1', (1, 2): 'B1',
    }
    # Track 1 stays near the origin; track 2 stays near (100,100).
    tracks_df = pd.DataFrame([
        {'track_id': 1, 't': 0, 'y': 1.0, 'x': 1.0, 'parent_track_id': -1},
        {'track_id': 1, 't': 1, 'y': 2.0, 'x': 2.0, 'parent_track_id': -1},
        {'track_id': 2, 't': 0, 'y': 98.0, 'x': 98.0, 'parent_track_id': -1},
        {'track_id': 2, 't': 1, 'y': 99.0, 'x': 99.0, 'parent_track_id': -1},
    ])
    conns = tracks_df_to_connections(tracks_df, label_centroid, label_to_id, 'ds', [])
    pairs = {(c['parentId'], c['childId']) for c in conns}
    assert pairs == {('A0', 'A1'), ('B0', 'B1')}


def test_compute_no_tag_raises():
    params = {
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'tags': [],
        'workerInterface': {'Tag of objects to track': []},
    }
    with pytest.raises(ValueError, match="No tag specified"):
        compute('ds', 'http://api', 'token', params)


@patch('entrypoint.run_ultrack')
@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_end_to_end(mock_ann_client_cls, mock_tile_cls, mock_run,
                            square_annotations):
    ann_client = mock_ann_client_cls.return_value
    ann_client.getAnnotationsByDatasetId.return_value = square_annotations
    ann_client.createMultipleConnections.return_value = {}

    tile_client = mock_tile_cls.return_value
    tile_client.tiles = {'sizeX': 20, 'sizeY': 20}

    # Ultrack returns one track linking the three squares. Centroids match the
    # rasterized square centroids (row=y, col=x): (3.5,3.5),(5.5,5.5),(7.5,7.5).
    def fake_run(masks, max_distance, allow_division, working_dir):
        return pd.DataFrame([
            {'track_id': 1, 't': 0, 'y': 3.5, 'x': 3.5, 'parent_track_id': -1},
            {'track_id': 1, 't': 1, 'y': 5.5, 'x': 5.5, 'parent_track_id': -1},
            {'track_id': 1, 't': 2, 'y': 7.5, 'x': 7.5, 'parent_track_id': -1},
        ])
    mock_run.side_effect = fake_run

    params = {
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'tags': ['Ultrack'],
        'workerInterface': {
            'Tag of objects to track': ['cell'],
            'Max distance': 50,
            'Allow divisions': True,
            'Batch XY': '',
            'Batch Z': '',
        },
    }

    compute('ds', 'http://api', 'token', params)

    ann_client.createMultipleConnections.assert_called_once()
    conns = ann_client.createMultipleConnections.call_args[0][0]
    pairs = {(c['parentId'], c['childId']) for c in conns}
    assert pairs == {('t0', 't1'), ('t1', 't2')}
    mock_run.assert_called_once()


@patch('entrypoint.run_ultrack')
@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_stamps_descriptive_tag_when_no_output_tags(
        mock_ann_client_cls, mock_tile_cls, mock_run, square_annotations):
    """Connections carry an 'Ultrack' tag even when no output tags are set."""
    ann_client = mock_ann_client_cls.return_value
    ann_client.getAnnotationsByDatasetId.return_value = square_annotations
    tile_client = mock_tile_cls.return_value
    tile_client.tiles = {'sizeX': 20, 'sizeY': 20}

    def fake_run(masks, max_distance, allow_division, working_dir):
        return pd.DataFrame([
            {'track_id': 1, 't': 0, 'y': 3.5, 'x': 3.5, 'parent_track_id': -1},
            {'track_id': 1, 't': 1, 'y': 5.5, 'x': 5.5, 'parent_track_id': -1},
            {'track_id': 1, 't': 2, 'y': 7.5, 'x': 7.5, 'parent_track_id': -1},
        ])
    mock_run.side_effect = fake_run

    params = {
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'tags': [],  # no output tags configured
        'workerInterface': {
            'Tag of objects to track': ['cell'],
            'Max distance': 50,
            'Allow divisions': True,
            'Batch XY': '',
            'Batch Z': '',
        },
    }
    compute('ds', 'http://api', 'token', params)

    conns = ann_client.createMultipleConnections.call_args[0][0]
    assert conns, "expected at least one connection"
    assert all('Ultrack' in c['tags'] for c in conns)


@patch('entrypoint.run_ultrack')
@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_no_annotations(mock_ann_client_cls, mock_tile_cls, mock_run):
    ann_client = mock_ann_client_cls.return_value
    ann_client.getAnnotationsByDatasetId.return_value = []

    params = {
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'tags': ['Ultrack'],
        'workerInterface': {
            'Tag of objects to track': ['cell'],
            'Batch XY': '',
            'Batch Z': '',
        },
    }
    compute('ds', 'http://api', 'token', params)
    ann_client.createMultipleConnections.assert_not_called()
    mock_run.assert_not_called()
