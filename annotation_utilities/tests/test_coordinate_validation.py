import sys
from pathlib import Path

package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_root))

from annotation_utilities import coordinate_validation as cv


# ---------------------------------------------------------------------------
# find_out_of_range
# ---------------------------------------------------------------------------

def test_all_in_range_returns_empty():
    assert cv.find_out_of_range({'IndexXY': 5}, xys=[0, 1, 4]) == {}


def test_xy_out_of_range_detected_with_size():
    # size 5 => valid 0-indexed 0..4; 5 and 6 are out of range
    assert cv.find_out_of_range({'IndexXY': 5}, xys=[4, 5, 6]) == {
        'XY': ([5, 6], 5)
    }


def test_missing_index_range_defaults_size_to_one():
    # No IndexRange at all => every dimension has size 1 (only coord 0 valid)
    assert cv.find_out_of_range({}, xys=[0]) == {}
    assert cv.find_out_of_range({}, xys=[1]) == {'XY': ([1], 1)}


def test_missing_subkey_defaults_that_dimension_to_one():
    # IndexXY present but IndexZ absent => Z size defaults to 1
    index_range = {'IndexXY': 3}
    assert cv.find_out_of_range(index_range, zs=[0]) == {}
    assert cv.find_out_of_range(index_range, zs=[1]) == {'Z': ([1], 1)}


def test_negative_coordinate_is_out_of_range():
    assert cv.find_out_of_range({'IndexXY': 5}, xys=[-1]) == {'XY': ([-1], 5)}


def test_multiple_dimensions_reported_together():
    index_range = {'IndexXY': 2, 'IndexZ': 2, 'IndexT': 2}
    result = cv.find_out_of_range(index_range, xys=[2], zs=[0], times=[5, 6])
    assert result == {'XY': ([2], 2), 'Time': ([5, 6], 2)}


def test_none_dimensions_are_skipped():
    # zs is None => never checked, even though it would be out of range
    assert cv.find_out_of_range({'IndexXY': 5}, xys=[0]) == {}


def test_bad_values_are_sorted_and_deduplicated():
    assert cv.find_out_of_range({'IndexXY': 5}, xys=[6, 5, 6, 5]) == {
        'XY': ([5, 6], 5)
    }


def test_empty_requested_list_is_in_range():
    assert cv.find_out_of_range({'IndexXY': 1}, xys=[]) == {}


# ---------------------------------------------------------------------------
# format_out_of_range_message
# ---------------------------------------------------------------------------

def test_message_single_contiguous_range_is_one_indexed():
    # 0-indexed 79..89 (the reported "80-90" input) => display 80-90
    invalid = {'XY': (list(range(79, 90)), 1)}
    message, info = cv.format_out_of_range_message(invalid)
    assert message == 'Batch XY out of range'
    assert info == 'Positions 80-90 do not exist'


def test_message_single_position_is_singular():
    invalid = {'Z': ([4], 1)}
    message, info = cv.format_out_of_range_message(invalid)
    assert message == 'Batch Z out of range'
    assert info == 'Position 5 does not exist'


def test_message_non_contiguous_positions_are_comma_listed():
    invalid = {'XY': ([4, 6], 5)}
    message, info = cv.format_out_of_range_message(invalid)
    assert message == 'Batch XY out of range'
    assert info == 'Positions 5, 7 do not exist'


def test_message_multiple_dimensions_combined():
    invalid = {'XY': ([79, 80], 1), 'Z': ([4], 1)}
    message, info = cv.format_out_of_range_message(invalid)
    assert message == 'Batch coordinates out of range'
    assert info == (
        'Batch XY: positions 80-81 do not exist; '
        'Batch Z: position 5 does not exist'
    )


def test_message_uses_correct_labels_for_each_dimension():
    _, info = cv.format_out_of_range_message({'Time': ([9], 5)})
    message, _ = cv.format_out_of_range_message({'Time': ([9], 5)})
    assert message == 'Batch Time out of range'
    assert info == 'Position 10 does not exist'
