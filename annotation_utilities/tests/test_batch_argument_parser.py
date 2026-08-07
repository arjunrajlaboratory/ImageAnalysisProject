import pytest

from annotation_utilities import batch_argument_parser


BATCH_FIELD_NAMES = ('Batch XY', 'Batch Z', 'Batch Time')


def test_batch_interface_fields_returns_the_three_standard_fields():
    fields = batch_argument_parser.batch_interface_fields()

    assert tuple(fields) == BATCH_FIELD_NAMES
    assert all(f['type'] == 'text' for f in fields.values())


def test_batch_interface_fields_advertises_all_in_every_placeholder():
    """The placeholder is a promise about process_range_list(); the two live in
    one module so they cannot drift, and this pins the promise."""
    fields = batch_argument_parser.batch_interface_fields()

    for name in BATCH_FIELD_NAMES:
        placeholder = fields[name]['vueAttrs']['placeholder']
        assert placeholder == batch_argument_parser.BATCH_RANGE_PLACEHOLDER
        assert 'or all' in placeholder


def test_batch_interface_fields_numbers_display_order_from_the_start():
    fields = batch_argument_parser.batch_interface_fields(display_order=9)

    assert [fields[n]['displayOrder'] for n in BATCH_FIELD_NAMES] == [9, 10, 11]


def test_batch_interface_fields_verb_reaches_every_label():
    fields = batch_argument_parser.batch_interface_fields(verb='process')

    labels = [fields[n]['vueAttrs']['label'] for n in BATCH_FIELD_NAMES]
    assert labels == [
        'Enter the XY positions you want to process',
        'Enter the Z slices you want to process',
        'Enter the Time points you want to process',
    ]


def test_batch_interface_fields_tooltips_can_be_omitted():
    with_tips = batch_argument_parser.batch_interface_fields()
    without = batch_argument_parser.batch_interface_fields(tooltips=False)

    assert all('tooltip' in f['vueAttrs'] for f in with_tips.values())
    assert not any('tooltip' in f['vueAttrs'] for f in without.values())


def test_batch_interface_fields_returns_independent_dicts():
    """Workers mutate the result (sam2_propagate rewords two labels), so a
    shared/cached dict would leak one worker's wording into another's."""
    first = batch_argument_parser.batch_interface_fields()
    first['Batch Z']['vueAttrs']['label'] = 'mutated'

    assert batch_argument_parser.batch_interface_fields(
        )['Batch Z']['vueAttrs']['label'] != 'mutated'


def test_batch_interface_fields_are_parseable_by_get_batch_ranges():
    """The helper's field names must be exactly the ones get_batch_ranges reads;
    a rename in one without the other silently stops batching."""
    fields = batch_argument_parser.batch_interface_fields()
    worker_interface = {name: 'all' for name in fields}

    result = batch_argument_parser.get_batch_ranges(
        tile={'XY': 0, 'Z': 0, 'Time': 0},
        worker_interface=worker_interface,
        index_range={'IndexXY': 2, 'IndexZ': 3, 'IndexT': 1},
    )

    assert result == ([0, 1], [0, 1, 2], [0])


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
