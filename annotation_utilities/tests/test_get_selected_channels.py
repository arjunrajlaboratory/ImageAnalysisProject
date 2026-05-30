import sys
from pathlib import Path

import pytest


package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_root))

from annotation_utilities.annotation_tools import get_selected_channels


def test_dict_returns_sorted_selected_channels():
    assert get_selected_channels({'0': True, '1': False}) == [0]
    assert get_selected_channels({'2': True, '0': True, '1': False}) == [0, 2]


def test_empty_dict_returns_empty_list():
    assert get_selected_channels({}) == []


def test_none_returns_empty_list():
    assert get_selected_channels(None) == []


def test_all_false_dict_returns_empty_list():
    assert get_selected_channels({'0': False, '1': False}) == []


@pytest.mark.parametrize("bad_value", [[0], [], [0, 1], (0,), '0', 0, True])
def test_non_dict_values_raise_value_error(bad_value):
    # The list form ([0]) in particular has been seen in the wild; it must be
    # rejected rather than silently recovered.
    with pytest.raises(ValueError):
        get_selected_channels(bad_value)


def test_error_message_includes_field_name():
    with pytest.raises(ValueError, match="Channels to correct"):
        get_selected_channels([0], field_name='Channels to correct')
