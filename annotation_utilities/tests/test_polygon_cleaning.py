import sys
from pathlib import Path

from shapely.geometry import Polygon

package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_root))

from annotation_utilities.annotation_tools import (
    geometry_to_polygon_coords,
    polygons_to_annotations,
)


# ---------------------------------------------------------------------------
# geometry_to_polygon_coords helper
# ---------------------------------------------------------------------------

def test_helper_keeps_valid_polygon():
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    result = geometry_to_polygon_coords(square)
    assert len(result) == 1
    assert len(result[0]) >= 4


def test_helper_drops_empty_geometry_from_negative_buffer():
    tiny = Polygon([(0, 0), (1.5, 0), (1.5, 1.5), (0, 1.5)]).buffer(-1.0)
    assert tiny.is_empty
    assert geometry_to_polygon_coords(tiny) == []


def test_helper_drops_zero_area_polygon():
    sliver = Polygon([(0, 0), (1, 1), (2, 2)])  # colinear -> zero area
    assert not sliver.is_empty
    assert geometry_to_polygon_coords(sliver) == []


def test_helper_expands_multipolygon_from_negative_buffer():
    dumbbell = Polygon([
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 5), (10, 5), (10, 9),
        (0, 9), (0, 5), (4, 5), (4, 4), (0, 4),
    ]).buffer(-1.0)
    assert dumbbell.geom_type == "MultiPolygon"
    result = geometry_to_polygon_coords(dumbbell)
    assert len(result) == 2
    assert all(len(coords) >= 4 for coords in result)


def test_helper_handles_none():
    assert geometry_to_polygon_coords(None) == []


# ---------------------------------------------------------------------------
# polygons_to_annotations defense-in-depth (SAM2 chokepoint)
# ---------------------------------------------------------------------------

def test_polygons_to_annotations_skips_empty_polygon():
    valid = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    empty = Polygon([(0, 0), (1.5, 0), (1.5, 1.5), (0, 1.5)]).buffer(-1.0)

    annotations = polygons_to_annotations([valid, empty], "ds")

    assert len(annotations) == 1
    assert len(annotations[0]["coordinates"]) > 0


def test_polygons_to_annotations_expands_multipolygon():
    dumbbell = Polygon([
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 5), (10, 5), (10, 9),
        (0, 9), (0, 5), (4, 5), (4, 4), (0, 4),
    ]).buffer(-1.0)
    assert dumbbell.geom_type == "MultiPolygon"

    annotations = polygons_to_annotations([dumbbell], "ds")

    assert len(annotations) == 2
    assert all(len(a["coordinates"]) > 0 for a in annotations)


def test_polygons_to_annotations_returns_empty_when_all_degenerate():
    empty = Polygon([(0, 0), (1.5, 0), (1.5, 1.5), (0, 1.5)]).buffer(-1.0)
    assert polygons_to_annotations([empty], "ds") == []


def test_polygons_to_annotations_preserves_xy_swap_and_drops_closing_point():
    # Existing behaviour: x/y are swapped and the duplicated closing point dropped.
    poly = Polygon([(1, 2), (3, 2), (3, 5), (1, 5)])
    annotations = polygons_to_annotations([poly], "ds", XY=4, Time=5, Z=6,
                                          tags=["t"], channel=2)
    assert len(annotations) == 1
    coords = annotations[0]["coordinates"]
    assert len(coords) == 4  # closing point excluded
    assert coords[0] == {"x": 2.0, "y": 1.0}  # swapped from (1, 2)
    assert annotations[0]["location"] == {"XY": 4, "Time": 5, "Z": 6}
    assert annotations[0]["tags"] == ["t"]
    assert annotations[0]["channel"] == 2
