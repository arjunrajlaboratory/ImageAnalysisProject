import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_root))

from annotation_utilities.annotation_tools import (
    clean_polygon_coords,
    geometry_to_polygon_coords,
    polygons_to_annotations,
    safe_buffer,
    safe_polygon,
    safe_simplify,
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


def test_helper_drops_invalid_positive_area_polygon():
    # Self-intersecting outline: invalid but with positive area, so the
    # zero-area filter alone would NOT catch it -- is_valid must.
    bad = Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2, 3), (5, 3),
                   (5, 5), (0, 5), (0, 3), (3, 3), (3, 2), (0, 2)])
    assert not bad.is_valid and bad.area > 0
    assert geometry_to_polygon_coords(bad) == []


def test_helper_keep_largest_only_collapses_multipolygon():
    dumbbell = Polygon([
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 5), (10, 5), (10, 9),
        (0, 9), (0, 5), (4, 5), (4, 4), (0, 4),
    ]).buffer(-1.0)
    assert dumbbell.geom_type == "MultiPolygon"
    # Default mode expands to every piece...
    assert len(geometry_to_polygon_coords(dumbbell)) == 2
    # ...keep_largest_only collapses to the single largest piece.
    largest = geometry_to_polygon_coords(dumbbell, keep_largest_only=True)
    assert len(largest) == 1
    assert len(largest[0]) >= 4


def test_helper_keep_largest_only_flattens_nested_geometrycollection():
    from shapely.geometry import MultiPolygon, GeometryCollection
    big = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])      # area 16
    small = Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])  # area 4
    nested = GeometryCollection([MultiPolygon([big, small])])
    # keep_largest_only must recurse through the collection -> the largest leaf.
    result = geometry_to_polygon_coords(nested, keep_largest_only=True)
    assert len(result) == 1
    assert Polygon(result[0]).area == 16  # the big piece, not []
    # Default mode still flattens to both leaves.
    assert len(geometry_to_polygon_coords(nested)) == 2


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


def test_polygons_to_annotations_collapses_multipolygon_to_largest():
    # SAM2 callers (sam2_propagate/video) require one annotation per input mask,
    # so a MultiPolygon must collapse to its single largest piece -- NOT expand.
    dumbbell = Polygon([
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 5), (10, 5), (10, 9),
        (0, 9), (0, 5), (4, 5), (4, 4), (0, 4),
    ]).buffer(-1.0)
    assert dumbbell.geom_type == "MultiPolygon"

    annotations = polygons_to_annotations([dumbbell], "ds")

    assert len(annotations) == 1
    assert len(annotations[0]["coordinates"]) > 0


def test_polygons_to_annotations_skips_invalid_polygon():
    valid = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    bad = Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2, 3), (5, 3),
                   (5, 5), (0, 5), (0, 3), (3, 3), (3, 2), (0, 2)])
    assert not bad.is_valid

    annotations = polygons_to_annotations([valid, bad], "ds")

    assert len(annotations) == 1


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


# ---------------------------------------------------------------------------
# safe_polygon / safe_buffer / safe_simplify: guarding geometry *construction*
#
# The geometry_to_polygon_coords chokepoint above only helps once a shapely
# geometry exists. Segmentation workers build that geometry from raw mask
# contours, where construction itself can blow up and take the whole run with
# it: a 1-2 point contour raises ValueError, and a non-finite coordinate makes
# simplify() raise GEOSException.
# ---------------------------------------------------------------------------

def test_safe_polygon_builds_a_valid_ring():
    poly = safe_polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert poly is not None
    assert poly.area == 100


def test_safe_polygon_rejects_contours_too_short_to_form_a_ring():
    # Polygon() raises ValueError: A linearring requires at least 4 coordinates.
    assert safe_polygon([(0, 0), (1, 1)]) is None
    assert safe_polygon([(0, 0)]) is None
    assert safe_polygon([]) is None


def test_safe_polygon_rejects_non_finite_coordinates():
    # These construct fine but explode later, inside simplify().
    assert safe_polygon([(0, 0), (np.nan, 5), (10, 10), (0, 10)]) is None
    assert safe_polygon([(0, 0), (np.inf, 5), (10, 10), (0, 10)]) is None


def test_safe_polygon_rejects_non_coordinate_input():
    assert safe_polygon(None) is None
    assert safe_polygon("not coordinates") is None
    assert safe_polygon([{"x": 1, "y": 2}]) is None


def test_safe_polygon_accepts_numpy_contours():
    contour = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
    poly = safe_polygon(contour)
    assert poly is not None
    assert poly.area == 64


def test_safe_polygon_ignores_a_third_coordinate_column():
    # A 3D ring would yield 3-tuples, which crash the (x, y) unpacking used to
    # build annotation coordinates.
    poly = safe_polygon([(0, 0, 5), (10, 0, 5), (10, 10, 5), (0, 10, 5)])
    assert poly is not None
    assert all(len(point) == 2 for point in poly.exterior.coords)


def test_safe_polygon_passes_through_an_existing_geometry():
    square = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    assert safe_polygon(square) is square


def test_safe_buffer_and_simplify_tolerate_none():
    assert safe_buffer(None, -1) is None
    assert safe_simplify(None, 0.7) is None


def test_safe_simplify_survives_geos_failure_on_non_finite_geometry():
    # Bypasses safe_polygon to prove the transform guard stands on its own:
    # this raises GEOSException (Non-finite envelope bounds) unguarded.
    nan_poly = Polygon([(0, 0), (np.nan, 5), (10, 10), (0, 10)])
    assert safe_simplify(nan_poly, 0.7) is None


def test_safe_buffer_returns_empty_geometry_rather_than_raising():
    tiny = Polygon([(0, 0), (1.5, 0), (1.5, 1.5), (0, 1.5)])
    eroded = safe_buffer(tiny, -1.0)
    # Empty is fine here -- geometry_to_polygon_coords drops it downstream.
    assert eroded is None or eroded.is_empty


# ---------------------------------------------------------------------------
# clean_polygon_coords: contour -> annotation-ready rings, in one guarded step
# ---------------------------------------------------------------------------

def test_clean_polygon_coords_applies_padding_and_smoothing():
    square = [(0, 0), (20, 0), (20, 20), (0, 20)]
    rings = clean_polygon_coords(square, padding=-1.0, smoothing=0.7)
    assert len(rings) == 1
    assert Polygon(rings[0]).area < 400  # eroded by the negative padding


def test_clean_polygon_coords_drops_unconstructable_contour():
    assert clean_polygon_coords([(0, 0), (1, 1)], padding=-1.0, smoothing=0.7) == []


def test_clean_polygon_coords_drops_non_finite_contour():
    nan_contour = [(0, 0), (np.nan, 5), (10, 10), (0, 10)]
    assert clean_polygon_coords(nan_contour, padding=-1.0, smoothing=0.7) == []


def test_clean_polygon_coords_drops_contour_eroded_away():
    tiny = [(0, 0), (1.5, 0), (1.5, 1.5), (0, 1.5)]
    assert clean_polygon_coords(tiny, padding=-1.0) == []


def test_clean_polygon_coords_expands_multipolygon_into_separate_rings():
    dumbbell = [
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 5), (10, 5), (10, 9),
        (0, 9), (0, 5), (4, 5), (4, 4), (0, 4),
    ]
    rings = clean_polygon_coords(dumbbell, padding=-1.0)
    assert len(rings) == 2
    assert all(len(ring) >= 4 for ring in rings)


def test_clean_polygon_coords_can_collapse_to_the_largest_piece():
    dumbbell = [
        (0, 0), (10, 0), (10, 4), (6, 4), (6, 5), (10, 5), (10, 9),
        (0, 9), (0, 5), (4, 5), (4, 4), (0, 4),
    ]
    rings = clean_polygon_coords(dumbbell, padding=-1.0, keep_largest_only=True)
    assert len(rings) == 1


def test_clean_polygon_coords_without_transforms_returns_the_ring():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    rings = clean_polygon_coords(square)
    assert len(rings) == 1
    assert Polygon(rings[0]).area == 100
