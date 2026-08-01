"""Tests for annotation_tools.split_channel_selection.

A channel selection is validated for *shape* by get_selected_channels, but its
indices are only meaningful against a particular dataset. A saved tool config
selecting channel 1, run against a single-channel dataset, parses cleanly and then
matches no frame, so the worker uploads an untouched copy of its input and reports
success. split_channel_selection is what lets a worker tell the two cases apart.
"""

import sys
from pathlib import Path

import pytest


package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_root))

from annotation_utilities.annotation_tools import split_channel_selection


def test_all_channels_present():
    assert split_channel_selection([0, 1], 2) == ([0, 1], [])


def test_selection_entirely_outside_range():
    """The production case: channel 1 selected on a single-channel dataset."""
    assert split_channel_selection([1], 1) == ([], [1])


def test_selection_partially_outside_range():
    assert split_channel_selection([0, 3], 2) == ([0], [3])


def test_empty_selection_is_not_reported_as_missing():
    """"Nothing selected" must stay distinguishable from "nothing selected exists"."""
    assert split_channel_selection([], 2) == ([], [])


def test_single_channel_dataset_accepts_channel_zero():
    assert split_channel_selection([0], 1) == ([0], [])


def test_results_are_sorted_and_deduplicated():
    assert split_channel_selection([3, 1, 1, 0, 3], 2) == ([0, 1], [3])


def test_negative_indices_are_missing_not_present():
    """get_selected_channels rejects these, but the helper must not index with them."""
    assert split_channel_selection([-1, 0], 2) == ([0], [-1])


def test_accepts_any_iterable():
    assert split_channel_selection({2, 0}, 2) == ([0], [2])
    assert split_channel_selection(range(3), 2) == ([0, 1], [2])


def test_does_not_mutate_or_alias_the_input():
    selection = [0, 5]
    present, missing = split_channel_selection(selection, 2)
    present.append(99)
    assert selection == [0, 5]


@pytest.mark.parametrize('num_channels', [0, -1])
def test_rejects_non_positive_channel_count(num_channels):
    """Every dataset has at least one channel; 0 would reject every selection."""
    with pytest.raises(ValueError):
        split_channel_selection([0], num_channels)


def test_rejects_non_integer_channel_count():
    with pytest.raises(ValueError):
        split_channel_selection([0], None)
    with pytest.raises(ValueError):
        split_channel_selection([0], '2')
