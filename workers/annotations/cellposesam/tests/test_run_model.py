"""Unit tests for Cellpose-SAM's polygon post-processing (``run_model``).

A production run failed with HTTP 400 ``data.coordinates must contain at least
1 items``, which aborted the upload of *all* 442 annotations found in the frame.
The upload chokepoint (``worker_client.create_polygon_annotations``) now drops
degenerate geometry, but ``run_model`` itself builds shapely polygons *before*
that chokepoint, and those constructions were unguarded:

* a 1-2 point contour (a one-pixel-wide mask) makes ``Polygon()`` raise
  ``ValueError: A linearring requires at least 4 coordinates``;
* a contour carrying a non-finite coordinate makes ``.simplify()`` raise
  ``GEOSException: Non-finite envelope bounds``.

Either one kills the whole run, so a single bad mask out of hundreds loses every
good annotation in the frame. These tests pin that one bad mask is skipped and
the good ones still come through.

``deeptile``/``cellpose`` are stubbed so this runs in the lightweight local venv
without the GPU worker stack. Run with:

    .cache/testvenv/bin/pytest workers/annotations/cellposesam/tests -q
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon


WORKER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = WORKER_DIR.parents[2]


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------

def _stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def entrypoint(monkeypatch):
    """Load cellposesam's entrypoint with its heavy dependencies stubbed out."""
    # The in-repo copies of the shared packages, so the test exercises the code
    # under review rather than whatever happens to be pip-installed.
    monkeypatch.syspath_prepend(str(REPO_ROOT / 'annotation_utilities'))
    monkeypatch.syspath_prepend(str(REPO_ROOT / 'worker_client'))
    monkeypatch.syspath_prepend(str(WORKER_DIR))

    # annotation_client is a server-side package; only interface()/compute()
    # touch it, not run_model.
    _stub_module(monkeypatch, 'annotation_client')
    _stub_module(monkeypatch, 'annotation_client.workers',
                 UPennContrastWorkerPreviewClient=lambda **kwargs: None)
    _stub_module(monkeypatch, 'annotation_client.utils',
                 sendError=lambda *a, **k: None,
                 sendWarning=lambda *a, **k: None,
                 sendProgress=lambda *a, **k: None)
    _stub_module(monkeypatch, 'annotation_client.annotations',
                 UPennContrastAnnotationClient=lambda **kwargs: None)
    _stub_module(monkeypatch, 'annotation_client.tiles',
                 UPennContrastDataset=lambda **kwargs: None)
    for name in ('worker_client', 'worker_client.worker_client'):
        monkeypatch.delitem(sys.modules, name, raising=False)

    # deeptile is imported lazily inside run_model; stub the tiling away so the
    # test drives the polygon post-processing directly.
    stitch = _stub_module(monkeypatch, 'deeptile.extensions.stitch',
                          stitch_polygons=lambda polygons: polygons)
    extensions = _stub_module(monkeypatch, 'deeptile.extensions', stitch=stitch)
    _stub_module(monkeypatch, 'deeptile',
                 load=lambda image: types.SimpleNamespace(
                     get_tiles=lambda **kwargs: image),
                 extensions=extensions)

    spec = importlib.util.spec_from_file_location(
        'cellposesam_entrypoint', WORKER_DIR / 'entrypoint.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'cellposesam_entrypoint', module)
    spec.loader.exec_module(module)
    return module


def _run(entrypoint, contours, padding=0.0, smoothing=0.0):
    """Run the post-processing pipeline over a fixed list of mask contours."""
    return entrypoint.run_model(
        image=np.zeros((4, 4), dtype=np.float32),
        cellpose=lambda tiles: contours,
        tile_size=1024,
        tile_overlap=0.1,
        padding=padding,
        smoothing=smoothing,
    )


# A comfortably large, healthy cell outline that survives padding/smoothing.
HEALTHY_CELL = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]


def _assert_annotation_ready(rings):
    """Every returned ring must be uploadable as a polygon annotation."""
    for ring in rings:
        coords = list(ring)
        # An empty coordinate list is exactly what the server rejects with 400.
        assert len(coords) >= 4
        for point in coords:
            x, y = point  # must unpack as 2D, or annotation building crashes
            assert np.isfinite(x) and np.isfinite(y)


# ---------------------------------------------------------------------------
# The reported failure: one bad mask must not lose the whole frame
# ---------------------------------------------------------------------------

def test_two_point_contour_does_not_abort_the_frame(entrypoint):
    """A 1px-wide mask cannot form a ring; skip it, keep the real cells."""
    two_points = [(5.0, 5.0), (5.0, 6.0)]

    rings = _run(entrypoint, [HEALTHY_CELL, two_points, HEALTHY_CELL],
                 padding=-1.0, smoothing=0.7)

    assert len(rings) == 2
    _assert_annotation_ready(rings)


def test_single_point_contour_does_not_abort_the_frame(entrypoint):
    single_point = [(5.0, 5.0)]

    rings = _run(entrypoint, [HEALTHY_CELL, single_point], padding=-1.0, smoothing=0.7)

    assert len(rings) == 1
    _assert_annotation_ready(rings)


def test_non_finite_contour_does_not_abort_the_frame(entrypoint):
    """A NaN coordinate makes simplify() raise GEOSException; skip that mask."""
    nan_contour = [(0.0, 0.0), (np.nan, 5.0), (10.0, 10.0), (0.0, 10.0)]

    rings = _run(entrypoint, [HEALTHY_CELL, nan_contour], padding=-1.0, smoothing=0.7)

    assert len(rings) == 1
    _assert_annotation_ready(rings)


def test_smoothing_only_survives_degenerate_contours(entrypoint):
    """Padding off, smoothing on -- the default interface configuration."""
    rings = _run(entrypoint, [HEALTHY_CELL, [(1.0, 1.0), (2.0, 2.0)]],
                 padding=0.0, smoothing=0.7)

    assert len(rings) == 1
    _assert_annotation_ready(rings)


def test_padding_only_survives_degenerate_contours(entrypoint):
    rings = _run(entrypoint, [HEALTHY_CELL, [(1.0, 1.0), (2.0, 2.0)]],
                 padding=2.0, smoothing=0.0)

    assert len(rings) == 1
    _assert_annotation_ready(rings)


def test_frame_of_only_degenerate_contours_returns_nothing(entrypoint):
    """Nothing valid must yield an empty list, never an empty-coordinate ring."""
    rings = _run(entrypoint, [[(1.0, 1.0), (2.0, 2.0)], [], [(0.0, 0.0)]],
                 padding=-1.0, smoothing=0.7)

    assert list(rings) == []


# ---------------------------------------------------------------------------
# Existing post-processing behavior must not regress
# ---------------------------------------------------------------------------

def test_negative_padding_drops_cells_eroded_to_nothing(entrypoint):
    """Small objects shrunk away by negative padding are dropped, not uploaded."""
    tiny = [(0.0, 0.0), (1.5, 0.0), (1.5, 1.5), (0.0, 1.5)]

    rings = _run(entrypoint, [HEALTHY_CELL, tiny], padding=-1.0, smoothing=0.0)

    assert len(rings) == 1
    _assert_annotation_ready(rings)


def test_negative_padding_splitting_a_cell_yields_both_pieces(entrypoint):
    """A pinched cell becomes a MultiPolygon; each piece is its own annotation."""
    dumbbell = [
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 5), (10, 5), (10, 9),
        (0, 9), (0, 5), (4, 5), (4, 4), (0, 4),
    ]
    assert Polygon(dumbbell).buffer(-1.0).geom_type == 'MultiPolygon'

    rings = _run(entrypoint, [dumbbell], padding=-1.0, smoothing=0.0)

    assert len(rings) == 2
    _assert_annotation_ready(rings)


def test_no_padding_or_smoothing_passes_contours_through(entrypoint):
    """With post-processing off, cellpose's own outlines reach the uploader."""
    rings = _run(entrypoint, [HEALTHY_CELL], padding=0.0, smoothing=0.0)

    assert len(rings) == 1
    assert list(rings[0]) == HEALTHY_CELL


def test_smoothing_reduces_vertex_count(entrypoint):
    """Smoothing must still actually simplify the outline."""
    jagged = [(0.0, 0.0), (10.0, 0.05), (20.0, 0.0), (20.0, 20.0),
              (10.0, 19.95), (0.0, 20.0)]

    rings = _run(entrypoint, [jagged], padding=0.0, smoothing=0.7)

    assert len(rings) == 1
    assert len(rings[0]) < len(jagged) + 1  # +1 for the closing vertex
