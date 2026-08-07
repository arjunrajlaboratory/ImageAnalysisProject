import importlib
import sys
import types
from pathlib import Path

import pytest


def _load_worker_client(monkeypatch, dataset_client, errors):
    package_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(package_root))

    annotation_client = types.ModuleType("annotation_client")
    annotations = types.ModuleType("annotation_client.annotations")
    tiles = types.ModuleType("annotation_client.tiles")
    utils = types.ModuleType("annotation_client.utils")

    annotations.UPennContrastAnnotationClient = lambda **kwargs: types.SimpleNamespace()
    tiles.UPennContrastDataset = lambda **kwargs: dataset_client
    utils.sendProgress = lambda *args, **kwargs: None
    utils.sendError = lambda message, info=None: errors.append((message, info))

    monkeypatch.setitem(sys.modules, "annotation_client", annotation_client)
    monkeypatch.setitem(sys.modules, "annotation_client.annotations", annotations)
    monkeypatch.setitem(sys.modules, "annotation_client.tiles", tiles)
    monkeypatch.setitem(sys.modules, "annotation_client.utils", utils)

    sys.modules.pop("worker_client", None)
    sys.modules.pop("worker_client.worker_client", None)
    return importlib.import_module("worker_client").WorkerClient


class DummyDatasetClient:
    def __init__(self, index_range):
        self.tiles = {"IndexRange": index_range}
        self.frame_requests = []

    def coordinatesToFrameIndex(self, xy, z, time, channel):
        self.frame_requests.append((xy, z, time, channel))
        return 0

    def getRegion(self, dataset_id, frame):
        raise AssertionError("Invalid coordinates must be rejected before image access")


def _params(worker_interface=None):
    return {
        "assignment": {},
        "channel": 0,
        "connectTo": {"tags": []},
        "tags": [],
        "tile": {"XY": 0, "Z": 0, "Time": 0},
        "workerInterface": worker_interface or {},
    }


def test_process_reports_zero_in_one_indexed_batch_z_before_image_access(monkeypatch):
    errors = []
    dataset_client = DummyDatasetClient({"IndexZ": 35})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, errors)
    worker = WorkerClient(
        "dataset", "http://api", "token", _params({"Batch Z": "0-34"}))

    with pytest.raises(ValueError, match="Batch Z"):
        worker.process(lambda image: [], f_annotation="polygon")

    assert dataset_client.frame_requests == []
    assert errors == [(
        "Batch range is out of bounds.",
        "Batch Z contains invalid position 0. Batch positions start at 1; this "
        "dataset has 35 Z positions, so its valid range is 1-35.",
    )]


def test_validate_coordinates_reports_upper_out_of_range_value(monkeypatch):
    errors = []
    dataset_client = DummyDatasetClient({"IndexXY": 2})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, errors)
    worker = WorkerClient(
        "dataset", "http://api", "token", _params({"Batch XY": "1-3"}))

    with pytest.raises(ValueError, match="Batch XY"):
        worker.validate_coordinates()

    assert errors == [(
        "Batch range is out of bounds.",
        "Batch XY contains invalid position 3. Batch positions start at 1; this "
        "dataset has 2 XY positions, so its valid range is 1-2.",
    )]


def test_validate_coordinates_defaults_missing_dimension_size_to_one(monkeypatch):
    errors = []
    dataset_client = DummyDatasetClient({})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, errors)
    worker = WorkerClient(
        "dataset", "http://api", "token", _params({"Batch Time": "2"}))

    with pytest.raises(ValueError, match="Batch Time"):
        worker.validate_coordinates()

    assert errors[0][1] == (
        "Batch Time contains invalid position 2. Batch positions start at 1; "
        "this dataset has 1 Time position, so its valid range is 1-1.")


def test_validate_coordinates_ignores_batch_for_stacked_dimension(monkeypatch):
    errors = []
    dataset_client = DummyDatasetClient({"IndexZ": 35})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, errors)
    worker = WorkerClient(
        "dataset", "http://api", "token", _params({"Batch Z": "0"}))

    worker.validate_coordinates(stack_zs="all")

    assert errors == []


def test_validate_coordinates_reports_invalid_stack_channel(monkeypatch):
    errors = []
    dataset_client = DummyDatasetClient({"IndexC": 1, "IndexZ": 35})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, errors)
    worker = WorkerClient("dataset", "http://api", "token", _params())

    with pytest.raises(ValueError, match="channel index 1"):
        worker.validate_coordinates(stack_channels=[1])

    assert errors == [(
        "Selected channels are outside this dataset.",
        "The selected channel index 1 does not exist in this dataset, which has "
        "1 channel (valid indices: 0-0).",
    )]


def test_validate_coordinates_does_not_consume_batch_ranges(monkeypatch):
    errors = []
    dataset_client = DummyDatasetClient({"IndexZ": 2})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, errors)
    worker = WorkerClient(
        "dataset", "http://api", "token", _params({"Batch Z": "1-2"}))

    worker.validate_coordinates()
    first_read = list(worker.batch_z)
    second_read = list(worker.batch_z)

    assert first_read == [0, 1]
    assert second_read == [0, 1]
    assert errors == []


def test_malformed_batch_range_reports_parse_error(monkeypatch):
    errors = []
    dataset_client = DummyDatasetClient({"IndexZ": 2})
    WorkerClient = _load_worker_client(monkeypatch, dataset_client, errors)

    with pytest.raises(ValueError, match="Batch Z must contain 1-based"):
        WorkerClient(
            "dataset", "http://api", "token",
            _params({"Batch Z": "first-third"}))

    assert errors[0][0] == "Could not read the batch range."
    assert "for example '1-3, 5-8'" in errors[0][1]
