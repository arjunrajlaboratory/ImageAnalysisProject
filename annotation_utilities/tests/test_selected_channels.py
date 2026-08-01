"""Tests for annotation_tools.get_selected_channels.

Regression coverage for a production crash in the cellposesam worker:

    AttributeError: 'list' object has no attribute 'items'

`channelCheckboxes` is documented to return {'0': True, '1': False}, but a saved
tool config held a bare list of selected indices ([0]) instead, so every worker
that called .items() on the raw value crashed before it could validate anything.

The list shape is rejected rather than normalized: the checkbox widget never
emitted it, so a config carrying it came from somewhere outside the UI, and
guessing which channel it meant risks running the tool on the wrong data.
"""

import sys
from pathlib import Path

import pytest


package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_root))

from annotation_utilities.annotation_tools import get_selected_channels


class TestDictForm:
    """The documented shape: {channel index: checked}."""

    def test_returns_checked_channels_only(self):
        assert get_selected_channels({'0': True, '1': False, '2': True}) == [0, 2]

    def test_empty_dict_is_no_selection(self):
        assert get_selected_channels({}) == []

    def test_nothing_checked_is_no_selection(self):
        assert get_selected_channels({'0': False, '1': False}) == []

    def test_result_is_sorted(self):
        assert get_selected_channels({'2': True, '0': True, '1': True}) == [0, 1, 2]

    def test_int_keys_are_accepted(self):
        assert get_selected_channels({0: True, 1: False, 3: True}) == [0, 3]

    def test_non_integer_key_is_rejected(self):
        with pytest.raises(ValueError):
            get_selected_channels({'DAPI': True})


    def test_duplicate_keys_are_collapsed(self):
        # '0' and 0 are distinct dict keys but the same channel.
        assert get_selected_channels({'0': True, 0: True, '1': True}) == [0, 1]

    def test_negative_index_is_rejected(self):
        with pytest.raises(ValueError):
            get_selected_channels({'-1': True})


class TestListFormIsRejected:
    """The shape from the crash: a bare list of selected indices."""

    def test_single_channel_list_is_rejected(self):
        # The exact payload from the reported crash. It very likely meant
        # "channel 0", but the config is malformed and must be re-saved rather
        # than silently interpreted.
        with pytest.raises(ValueError):
            get_selected_channels([0])

    def test_multi_channel_list_is_rejected(self):
        with pytest.raises(ValueError):
            get_selected_channels([2, 0])

    def test_empty_list_is_rejected(self):
        # An unset slot in a malformed config serializes as [] rather than {}.
        # It is still the wrong shape, so it fails alongside its siblings
        # instead of quietly reading as "nothing selected".
        with pytest.raises(ValueError):
            get_selected_channels([])

    def test_tuple_is_rejected(self):
        with pytest.raises(ValueError):
            get_selected_channels((1, 2))

    def test_list_of_check_states_is_rejected(self):
        with pytest.raises(ValueError):
            get_selected_channels([True, False])

    def test_error_message_explains_the_expected_shape(self):
        with pytest.raises(ValueError, match='mapping of channel index'):
            get_selected_channels([0], 'Channel for Slot 1')


class TestUnsetAndInvalid:
    def test_none_is_no_selection(self):
        assert get_selected_channels(None) == []

    def test_empty_string_is_no_selection(self):
        assert get_selected_channels('') == []

    def test_bare_int_is_rejected(self):
        with pytest.raises(ValueError):
            get_selected_channels(0)

    def test_non_empty_string_is_rejected(self):
        with pytest.raises(ValueError):
            get_selected_channels('0')

    def test_error_message_names_the_field(self):
        with pytest.raises(ValueError, match='Channel for Slot 1'):
            get_selected_channels(3.5, 'Channel for Slot 1')
