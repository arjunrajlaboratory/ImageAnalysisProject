import importlib
import sys
import types
from pathlib import Path

import numpy as np


def _load_worker_client(monkeypatch, dataset_client):
    package_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(package_root))

    annotation_client = types.ModuleType("annotation_client")
    annotations = types.ModuleType("annotation_client.annotations")
    tiles = types.ModuleType("annotation_client.tiles")
    utils = types.ModuleType("annotation_client.utils")

    annotations.UPennContrastAnnotationClient = lambda **kwargs: types.SimpleNamespace()
    tiles.UPennContrastDataset = lambda **kwargs: dataset_client
    utils.sendProgress = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "annotation_client", annotation_client)
    monkeypatch.setitem(sys.modules, "annotation_client.annotations", annotations)
    monkeypatch.setitem(sys.modules, "annotation_client.tiles", tiles)
    monkeypatch.setitem(sys.modules, "annotation_client.utils", utils)

    sys.modules.pop("worker_client", None)
    sys.modules.pop("worker_client.worker_client", None)
    return importlib.import_module("worker_client").WorkerClient


class DummyDatasetClient:
    def __init__(self, tiles):
        self.tiles = tiles
        self.frame_requests = []

    def coordinatesToFrameIndex(self, xy, z, time, channel):
        self.frame_requests.append((xy, z, time, channel))
        return len(self.frame_requests) - 1

    def getRegion(self, dataset_id, frame):
        return np.full((2, 2), frame, dtype=np.uint8)


def _params():
    return {
        "assignment": {},
        "channel": 0,
        "connectTo": {"tags": []},
        "tags": [],
        "tile": {"XY": 0, "Z": 0, "Time": 0},
        "workerInterface": {},
    }


def test_get_image_stack_defaults_missing_index_range_to_single_plane(monkeypatch):
    dataset_client = DummyDatasetClient(tiles={})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client)
    worker = WorkerClient("dataset", "http://api", "token", _params())

    stack = worker.get_image_stack((0, 0, 0, 0), stack_zs="all")

    assert stack.shape == (1, 2, 2)
    assert dataset_client.frame_requests == [(0, 0, 0, 0)]


def test_get_image_stack_defaults_missing_dimension_to_one(monkeypatch):
    dataset_client = DummyDatasetClient(tiles={"IndexRange": {"IndexT": 2}})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client)
    worker = WorkerClient("dataset", "http://api", "token", _params())

    stack = worker.get_image_stack((0, 0, 0, 0), stack_zs="all")

    assert stack.shape == (1, 2, 2)
    assert dataset_client.frame_requests == [(0, 0, 0, 0)]


def test_get_image_stack_uses_index_range_when_present(monkeypatch):
    dataset_client = DummyDatasetClient(tiles={"IndexRange": {"IndexC": 3}})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client)
    worker = WorkerClient("dataset", "http://api", "token", _params())

    stack = worker.get_image_stack((0, 0, 0, 0), stack_channels="all")

    assert stack.shape == (3, 2, 2)
    assert dataset_client.frame_requests == [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 0, 2),
    ]
