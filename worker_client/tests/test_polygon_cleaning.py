import importlib
import sys
import types
from pathlib import Path

from shapely.geometry import Polygon


def _load_worker_client_module(monkeypatch, annotation_client_obj=None):
    package_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(package_root))

    annotation_client = types.ModuleType("annotation_client")
    annotations = types.ModuleType("annotation_client.annotations")
    tiles = types.ModuleType("annotation_client.tiles")
    utils = types.ModuleType("annotation_client.utils")

    annotations.UPennContrastAnnotationClient = (
        lambda **kwargs: annotation_client_obj
        if annotation_client_obj is not None
        else types.SimpleNamespace()
    )
    tiles.UPennContrastDataset = lambda **kwargs: types.SimpleNamespace(tiles={})
    utils.sendProgress = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "annotation_client", annotation_client)
    monkeypatch.setitem(sys.modules, "annotation_client.annotations", annotations)
    monkeypatch.setitem(sys.modules, "annotation_client.tiles", tiles)
    monkeypatch.setitem(sys.modules, "annotation_client.utils", utils)

    sys.modules.pop("worker_client", None)
    sys.modules.pop("worker_client.worker_client", None)
    return importlib.import_module("worker_client")


class RecordingAnnotationClient:
    """Captures createMultipleAnnotations payloads instead of hitting the server."""

    def __init__(self):
        self.created = []

    def createMultipleAnnotations(self, annotation_list):
        self.created.append(annotation_list)
        return [{"_id": str(i)} for i in range(len(annotation_list))]

    def connectToNearest(self, *args, **kwargs):
        raise AssertionError("connectToNearest should not be called in these tests")


def _params():
    return {
        "assignment": {},
        "channel": 0,
        "connectTo": {"tags": []},
        "tags": ["blob"],
        "tile": {"XY": 0, "Z": 0, "Time": 0},
        "workerInterface": {},
    }


# ---------------------------------------------------------------------------
# geometry_to_polygon_coords helper
# ---------------------------------------------------------------------------

def test_helper_keeps_valid_polygon(monkeypatch):
    wc = _load_worker_client_module(monkeypatch)
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    result = wc.geometry_to_polygon_coords(square)
    assert len(result) == 1
    assert len(result[0]) >= 4


def test_helper_drops_empty_geometry_from_negative_buffer(monkeypatch):
    wc = _load_worker_client_module(monkeypatch)
    # A small object eroded to nothing by negative padding.
    tiny = Polygon([(0, 0), (1.5, 0), (1.5, 1.5), (0, 1.5)]).buffer(-1.0)
    assert tiny.is_empty
    assert wc.geometry_to_polygon_coords(tiny) == []


def test_helper_drops_zero_area_polygon(monkeypatch):
    wc = _load_worker_client_module(monkeypatch)
    # Colinear points construct a valid (4-coord) but zero-area ring.
    sliver = Polygon([(0, 0), (1, 1), (2, 2)])
    assert not sliver.is_empty
    assert wc.geometry_to_polygon_coords(sliver) == []


def test_helper_expands_multipolygon_from_negative_buffer(monkeypatch):
    wc = _load_worker_client_module(monkeypatch)
    # Dumbbell shape: negative buffer splits the thin neck into two pieces.
    dumbbell = Polygon([
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 5), (10, 5), (10, 9),
        (0, 9), (0, 5), (4, 5), (4, 4), (0, 4),
    ]).buffer(-1.0)
    assert dumbbell.geom_type == "MultiPolygon"
    result = wc.geometry_to_polygon_coords(dumbbell)
    assert len(result) == 2
    assert all(len(coords) >= 4 for coords in result)


def test_helper_handles_none(monkeypatch):
    wc = _load_worker_client_module(monkeypatch)
    assert wc.geometry_to_polygon_coords(None) == []


# ---------------------------------------------------------------------------
# create_polygon_annotations defense-in-depth
# ---------------------------------------------------------------------------

def test_create_polygon_annotations_skips_degenerate_polygons(monkeypatch):
    recorder = RecordingAnnotationClient()
    wc = _load_worker_client_module(monkeypatch, recorder)
    worker = wc.WorkerClient("ds", "http://api", "tok", _params())

    valid = [(0, 0), (10, 0), (10, 10), (0, 10)]
    empty = []  # Polygon([]) -> empty geometry

    worker.create_polygon_annotations((0, 0, 0, 0), [valid, empty])

    assert len(recorder.created) == 1
    uploaded = recorder.created[0]
    assert len(uploaded) == 1
    assert len(uploaded[0]["coordinates"]) >= 4
    assert all(len(a["coordinates"]) > 0 for a in uploaded)


def test_create_polygon_annotations_skips_unconstructable_polygons(monkeypatch):
    recorder = RecordingAnnotationClient()
    wc = _load_worker_client_module(monkeypatch, recorder)
    worker = wc.WorkerClient("ds", "http://api", "tok", _params())

    valid = [(0, 0), (10, 0), (10, 10), (0, 10)]
    two_points = [(0, 0), (1, 1)]  # Polygon() raises ValueError on < 4 coords

    worker.create_polygon_annotations((0, 0, 0, 0), [valid, two_points])

    assert len(recorder.created) == 1
    assert len(recorder.created[0]) == 1


def test_create_polygon_annotations_no_upload_when_all_degenerate(monkeypatch):
    recorder = RecordingAnnotationClient()
    wc = _load_worker_client_module(monkeypatch, recorder)
    worker = wc.WorkerClient("ds", "http://api", "tok", _params())

    worker.create_polygon_annotations((0, 0, 0, 0), [[], [(0, 0), (1, 1), (2, 2)]])

    # Nothing valid -> never POST (an empty coordinates payload would 400).
    assert recorder.created == []
