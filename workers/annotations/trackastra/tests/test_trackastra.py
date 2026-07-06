import numpy as np
import networkx as nx
import pytest
from unittest.mock import patch, MagicMock

from entrypoint import (
    interface,
    compute,
    group_annotations_by_location,
    build_label_stack,
    track_graph_to_connections,
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
    assert 'Channel' in interface_data
    assert 'Model' in interface_data
    assert 'Tracking mode' in interface_data
    assert interface_data['Tag of objects to track']['type'] == 'tags'
    assert interface_data['Channel']['type'] == 'channel'


def test_group_annotations_by_location():
    anns = [
        {'shape': 'polygon', 'coordinates': [1, 2, 3], 'location': {'XY': 0, 'Z': 0, 'Time': 0}},
        {'shape': 'polygon', 'coordinates': [1, 2, 3], 'location': {'XY': 0, 'Z': 0, 'Time': 1}},
        {'shape': 'polygon', 'coordinates': [1, 2, 3], 'location': {'XY': 1, 'Z': 0, 'Time': 0}},
        # Too few coordinates -> dropped
        {'shape': 'polygon', 'coordinates': [1, 2], 'location': {'XY': 0, 'Z': 0, 'Time': 2}},
        # Wrong shape -> dropped
        {'shape': 'point', 'coordinates': [1], 'location': {'XY': 0, 'Z': 0, 'Time': 0}},
    ]
    groups = group_annotations_by_location(anns)
    assert set(groups.keys()) == {(0, 0), (1, 0)}
    assert len(groups[(0, 0)]) == 2
    assert len(groups[(1, 0)]) == 1


def test_build_label_stack(square_annotations):
    time_points = [0, 1, 2]
    masks, label_to_id, label_centroid = build_label_stack(
        square_annotations, time_points, height=20, width=20)

    assert masks.shape == (3, 20, 20)
    # One object per frame, each gets label 1.
    assert label_to_id[(0, 1)] == 't0'
    assert label_to_id[(1, 1)] == 't1'
    assert label_to_id[(2, 1)] == 't2'
    # The label pixels exist in each frame.
    for t_idx in range(3):
        assert (masks[t_idx] == 1).sum() > 0
    # Centroid of the first square (x 2..6, y 2..6) is near (4, 4) in (row, col).
    row, col = label_centroid[(0, 1)]
    assert abs(row - 3.5) < 1.5
    assert abs(col - 3.5) < 1.5


def test_build_label_stack_skips_short_polygons():
    anns = [
        {'_id': 'bad', 'shape': 'polygon',
         'coordinates': [{'x': 0, 'y': 0}, {'x': 1, 'y': 1}],
         'location': {'Time': 0, 'XY': 0, 'Z': 0}},
    ]
    masks, label_to_id, _ = build_label_stack(anns, [0], height=10, width=10)
    assert masks.sum() == 0
    assert label_to_id == {}


def test_track_graph_to_connections_linear():
    """A -> B -> C linear track becomes two connections."""
    g = nx.DiGraph()
    g.add_node(0, time=0, label=1)
    g.add_node(1, time=1, label=1)
    g.add_node(2, time=2, label=1)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    label_to_id = {(0, 1): 't0', (1, 1): 't1', (2, 1): 't2'}

    conns = track_graph_to_connections(g, label_to_id, 'ds', ['Trackastra'])
    pairs = {(c['parentId'], c['childId']) for c in conns}
    assert pairs == {('t0', 't1'), ('t1', 't2')}
    assert all(c['datasetId'] == 'ds' for c in conns)
    assert all(c['tags'] == ['Trackastra'] for c in conns)


def test_track_graph_to_connections_division():
    """A parent dividing into two daughters yields two connections."""
    g = nx.DiGraph()
    g.add_node(0, time=0, label=1)
    g.add_node(1, time=1, label=1)
    g.add_node(2, time=1, label=2)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    label_to_id = {(0, 1): 'parent', (1, 1): 'childA', (1, 2): 'childB'}

    conns = track_graph_to_connections(g, label_to_id, 'ds', ['Trackastra'])
    pairs = {(c['parentId'], c['childId']) for c in conns}
    assert pairs == {('parent', 'childA'), ('parent', 'childB')}


def test_track_graph_orients_by_time_not_edge_direction():
    """Even if an edge is stored later->earlier, parent is the earlier node."""
    g = nx.DiGraph()
    g.add_node(0, time=5, label=1)
    g.add_node(1, time=2, label=1)
    g.add_edge(0, 1)  # stored high-time -> low-time
    label_to_id = {(5, 1): 'late', (2, 1): 'early'}

    conns = track_graph_to_connections(g, label_to_id, 'ds', [])
    assert len(conns) == 1
    assert conns[0]['parentId'] == 'early'
    assert conns[0]['childId'] == 'late'


def test_track_graph_skips_unmapped_nodes():
    g = nx.DiGraph()
    g.add_node(0, time=0, label=1)
    g.add_node(1, time=1, label=99)  # not in label_to_id
    g.add_edge(0, 1)
    label_to_id = {(0, 1): 't0'}
    conns = track_graph_to_connections(g, label_to_id, 'ds', [])
    assert conns == []


def test_compute_no_tag_raises():
    params = {
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'tags': [],
        'channel': 0,
        'workerInterface': {
            'Tag of objects to track': [],
            'Channel': 0,
        },
    }
    with pytest.raises(ValueError, match="No tag specified"):
        compute('ds', 'http://api', 'token', params)


@patch('entrypoint.run_trackastra')
@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_end_to_end(mock_ann_client_cls, mock_tile_cls, mock_run,
                            square_annotations):
    ann_client = mock_ann_client_cls.return_value
    ann_client.getAnnotationsByDatasetId.return_value = square_annotations
    ann_client.createMultipleConnections.return_value = {}

    tile_client = mock_tile_cls.return_value
    tile_client.tiles = {'sizeX': 20, 'sizeY': 20}
    tile_client.coordinatesToFrameIndex.return_value = 0
    tile_client.getRegion.return_value = np.zeros((20, 20), dtype=np.float32)

    # Fake track graph: t0 -> t1 -> t2 (single object per frame, all label 1).
    g = nx.DiGraph()
    g.add_node(0, time=0, label=1)
    g.add_node(1, time=1, label=1)
    g.add_node(2, time=2, label=1)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    mock_run.return_value = g

    params = {
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'tags': ['Trackastra'],
        'channel': 0,
        'workerInterface': {
            'Tag of objects to track': ['cell'],
            'Channel': 0,
            'Model': 'general_2d',
            'Tracking mode': 'greedy',
            'Batch XY': '',
            'Batch Z': '',
        },
    }

    compute('ds', 'http://api', 'token', params)

    ann_client.createMultipleConnections.assert_called_once()
    conns = ann_client.createMultipleConnections.call_args[0][0]
    pairs = {(c['parentId'], c['childId']) for c in conns}
    assert pairs == {('t0', 't1'), ('t1', 't2')}
    # run_trackastra was called once (single XY/Z group).
    mock_run.assert_called_once()


@patch('entrypoint.run_trackastra')
@patch('annotation_client.tiles.UPennContrastDataset')
@patch('annotation_client.annotations.UPennContrastAnnotationClient')
def test_compute_no_annotations(mock_ann_client_cls, mock_tile_cls, mock_run):
    ann_client = mock_ann_client_cls.return_value
    ann_client.getAnnotationsByDatasetId.return_value = []

    params = {
        'tile': {'XY': 0, 'Z': 0, 'Time': 0},
        'tags': ['Trackastra'],
        'channel': 0,
        'workerInterface': {
            'Tag of objects to track': ['cell'],
            'Channel': 0,
            'Batch XY': '',
            'Batch Z': '',
        },
    }
    compute('ds', 'http://api', 'token', params)
    ann_client.createMultipleConnections.assert_not_called()
    mock_run.assert_not_called()
