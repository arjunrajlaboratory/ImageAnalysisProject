import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_worker_client(monkeypatch, dataset_client, send_error_calls=None):
    package_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(package_root))

    annotation_client = types.ModuleType("annotation_client")
    annotations = types.ModuleType("annotation_client.annotations")
    tiles = types.ModuleType("annotation_client.tiles")
    utils = types.ModuleType("annotation_client.utils")

    annotations.UPennContrastAnnotationClient = lambda **kwargs: types.SimpleNamespace()
    tiles.UPennContrastDataset = lambda **kwargs: dataset_client
    utils.sendProgress = lambda *args, **kwargs: None
    if send_error_calls is None:
        send_error_calls = []
    utils.sendError = lambda *args, **kwargs: send_error_calls.append((args, kwargs))

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


# ---------------------------------------------------------------------------
# Out-of-range coordinate validation
# ---------------------------------------------------------------------------

def _batch_params(**worker_interface):
    return {
        "assignment": {},
        "channel": 0,
        "connectTo": {"tags": []},
        "tags": [],
        "tile": {"XY": 0, "Z": 0, "Time": 0},
        "workerInterface": worker_interface,
    }


def test_process_raises_and_sends_error_on_out_of_range_xy(monkeypatch):
    dataset_client = DummyDatasetClient(tiles={"IndexRange": {"IndexXY": 1}})
    send_error_calls = []
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, send_error_calls)
    worker = WorkerClient("dataset", "http://api", "token",
                          _batch_params(**{"Batch XY": "80-90"}))

    with pytest.raises(ValueError):
        worker.process(lambda image: [], lambda location, output: None)

    assert len(send_error_calls) == 1
    assert send_error_calls[0][0][0] == "Batch XY out of range"
    assert send_error_calls[0][1]["info"] == "Positions 80-90 do not exist"
    # The processing loop must never run when validation fails.
    assert dataset_client.frame_requests == []


def test_process_succeeds_for_in_range_xy(monkeypatch):
    dataset_client = DummyDatasetClient(tiles={"IndexRange": {"IndexXY": 5}})
    send_error_calls = []
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, send_error_calls)
    worker = WorkerClient("dataset", "http://api", "token",
                          _batch_params(**{"Batch XY": "1-3"}))

    annotated = []
    worker.process(lambda image: image,
                   lambda location, output: annotated.append(location))

    assert send_error_calls == []
    # Batch XY 1-3 (1-indexed) -> 0-indexed 0, 1, 2 -> three iterations.
    assert dataset_client.frame_requests == [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)]
    assert annotated == [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)]


def test_process_errors_when_index_range_missing_and_coord_nonzero(monkeypatch):
    # No IndexRange => every dimension defaults to size 1 (only position 1 valid).
    dataset_client = DummyDatasetClient(tiles={})
    send_error_calls = []
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, send_error_calls)
    worker = WorkerClient("dataset", "http://api", "token",
                          _batch_params(**{"Batch XY": "2"}))

    with pytest.raises(ValueError):
        worker.process(lambda image: [], lambda location, output: None)

    assert len(send_error_calls) == 1
    assert dataset_client.frame_requests == []


def test_process_raises_and_sends_error_on_out_of_range_z(monkeypatch):
    # Exercise a non-XY dimension through process() to lock in per-dimension
    # wiring (Batch XY is left in range / unspecified).
    dataset_client = DummyDatasetClient(tiles={"IndexRange": {"IndexZ": 1}})
    send_error_calls = []
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, send_error_calls)
    worker = WorkerClient("dataset", "http://api", "token",
                          _batch_params(**{"Batch Z": "3-4"}))

    with pytest.raises(ValueError):
        worker.process(lambda image: [], lambda location, output: None)

    assert len(send_error_calls) == 1
    assert send_error_calls[0][0][0] == "Batch Z out of range"
    assert send_error_calls[0][1]["info"] == "Positions 3-4 do not exist"
    assert dataset_client.frame_requests == []


def test_process_reports_multiple_out_of_range_dimensions(monkeypatch):
    dataset_client = DummyDatasetClient(
        tiles={"IndexRange": {"IndexXY": 1, "IndexZ": 1}})
    send_error_calls = []
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, send_error_calls)
    worker = WorkerClient("dataset", "http://api", "token",
                          _batch_params(**{"Batch XY": "5", "Batch Z": "3-4"}))

    with pytest.raises(ValueError):
        worker.process(lambda image: [], lambda location, output: None)

    assert len(send_error_calls) == 1
    assert send_error_calls[0][0][0] == "Batch coordinates out of range"
    assert send_error_calls[0][1]["info"] == (
        "Batch XY: position 5 does not exist; "
        "Batch Z: positions 3-4 do not exist"
    )
    assert dataset_client.frame_requests == []


def test_validate_coordinates_raises_directly(monkeypatch):
    dataset_client = DummyDatasetClient(tiles={"IndexRange": {"IndexXY": 1}})
    send_error_calls = []
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, send_error_calls)
    worker = WorkerClient("dataset", "http://api", "token",
                          _batch_params(**{"Batch XY": "80-90"}))

    with pytest.raises(ValueError):
        worker.validate_coordinates()

    assert send_error_calls[0][1]["info"] == "Positions 80-90 do not exist"


def test_validate_coordinates_returns_none_when_in_range(monkeypatch):
    dataset_client = DummyDatasetClient(tiles={"IndexRange": {"IndexXY": 5}})
    send_error_calls = []
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, send_error_calls)
    worker = WorkerClient("dataset", "http://api", "token",
                          _batch_params(**{"Batch XY": "1-3"}))

    assert worker.validate_coordinates() is None
    assert send_error_calls == []


def test_validate_coordinates_skips_stacked_dimension(monkeypatch):
    # Batch XY is out of range, but when XY is STACKED (not batched) the worker
    # uses the current (valid) tile XY instead, so validation must not fire.
    dataset_client = DummyDatasetClient(tiles={"IndexRange": {"IndexXY": 1}})
    send_error_calls = []
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, send_error_calls)
    worker = WorkerClient("dataset", "http://api", "token",
                          _batch_params(**{"Batch XY": "80-90"}))

    assert worker.validate_coordinates(stack_xys=[0]) is None
    assert send_error_calls == []
