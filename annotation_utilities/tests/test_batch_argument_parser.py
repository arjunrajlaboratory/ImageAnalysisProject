import pytest

from annotation_utilities import batch_argument_parser


def test_process_range_list_expands_all_to_supplied_values():
    result = batch_argument_parser.process_range_list(
        "all",
        convert_one_to_zero_index=True,
        all_values=range(3),
    )

    assert list(result) == [0, 1, 2]


def test_process_range_list_accepts_case_and_whitespace_for_all():
    result = batch_argument_parser.process_range_list(
        "  ALL  ",
        all_values=[2, 4],
    )

    assert list(result) == [2, 4]


def test_process_range_list_requires_values_to_expand_all():
    try:
        batch_argument_parser.process_range_list("all")
    except ValueError as exc:
        assert "all_values" in str(exc)
    else:
        raise AssertionError("Expected 'all' without all_values to fail")


def test_get_batch_ranges_expands_each_all_dimension():
    result = batch_argument_parser.get_batch_ranges(
        tile={"XY": 1, "Z": 2, "Time": 3},
        worker_interface={
            "Batch XY": "all",
            "Batch Z": "ALL",
            "Batch Time": " all ",
        },
        index_range={"IndexXY": 2, "IndexZ": 3, "IndexT": 4},
    )

    assert result == ([0, 1], [0, 1, 2], [0, 1, 2, 3])


def test_get_batch_ranges_defaults_missing_dimensions_to_one():
    result = batch_argument_parser.get_batch_ranges(
        tile={"XY": 0, "Z": 0, "Time": 0},
        worker_interface={
            "Batch XY": "all",
            "Batch Z": "all",
            "Batch Time": "all",
        },
        index_range={},
    )

    assert result == ([0], [0], [0])


def test_get_batch_ranges_preserves_numeric_ranges_and_current_tile_defaults():
    result = batch_argument_parser.get_batch_ranges(
        tile={"XY": 4, "Z": 5, "Time": 6},
        worker_interface={
            "Batch XY": "1-2",
            "Batch Z": "3",
            "Batch Time": "",
        },
        index_range={"IndexXY": 7, "IndexZ": 7, "IndexT": 7},
    )

    assert result == ([0, 1], [2], [6])


def test_get_batch_ranges_names_the_field_that_failed_to_parse():
    """Direct batching loops (the SAM workers) surface this message, so it has
    to identify the offending field rather than leak a bare parser traceback."""
    with pytest.raises(ValueError) as excinfo:
        batch_argument_parser.get_batch_ranges(
            tile={"XY": 0, "Z": 0, "Time": 0},
            worker_interface={
                "Batch XY": "",
                "Batch Z": "first-third",
                "Batch Time": "",
            },
            index_range={"IndexZ": 3},
        )

    message = str(excinfo.value)
    assert message.startswith("Batch Z must contain 1-based")
    assert "'1-3, 5-8'" in message
    assert "'all'" in message
    assert "'first-third'" in message


def test_get_batch_ranges_returns_reiterable_lists():
    """process_range_list yields one-shot generators; callers such as
    WorkerClient.validate_coordinates read the coordinates before processing
    them, so an exhausted generator would silently process nothing."""
    batch_xy, _, _ = batch_argument_parser.get_batch_ranges(
        tile={"XY": 0, "Z": 0, "Time": 0},
        worker_interface={"Batch XY": "all"},
        index_range={"IndexXY": 3},
    )

    assert list(batch_xy) == [0, 1, 2]
    assert list(batch_xy) == [0, 1, 2]
