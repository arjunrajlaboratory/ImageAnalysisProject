"""Tests for annotation_tools.get_required_select.

Regression coverage for a production crash in the sam_fewshot_segmentation
worker:

    FileNotFoundError: [Errno 2] No such file or directory: '/None.pth'

The worker's saved tool config held ``"Model": null`` even though the
interface defines a ``select`` with a default, so the checkpoint path was
built from ``None`` and the job died deep inside SAM's model loader — after
the "Loading model" progress message, with no hint that the model selection
was the problem.

A null (or stale) select value is rejected with a clear message rather than
silently substituted with the interface default: the value in the saved
config is what the user believes the tool will run with, and guessing a
model changes the segmentation output.
"""

import pytest

from annotation_utilities.annotation_tools import get_required_select


class TestValidValues:
    def test_plain_string_passes_through(self):
        assert get_required_select('sam_vit_h_4b8939', 'Model') == 'sam_vit_h_4b8939'

    def test_value_in_allowed_values(self):
        assert get_required_select(
            'sam2.1_hiera_small.pt', 'Model',
            allowed_values=['sam2.1_hiera_small.pt', 'sam2.1_hiera_large.pt'],
        ) == 'sam2.1_hiera_small.pt'

    def test_allowed_values_accepts_any_container(self):
        assert get_required_select(
            'a', 'Model', allowed_values={'a': 'cfg_a', 'b': 'cfg_b'}) == 'a'


class TestMissingValues:
    """The production case: the saved config held null for the field."""

    def test_none_is_rejected(self):
        with pytest.raises(ValueError, match="Model"):
            get_required_select(None, 'Model')

    def test_empty_string_is_rejected(self):
        with pytest.raises(ValueError, match="Model"):
            get_required_select('', 'Model')

    def test_whitespace_string_is_rejected(self):
        with pytest.raises(ValueError, match="Model"):
            get_required_select('   ', 'Model')

    def test_error_message_says_how_to_fix(self):
        with pytest.raises(ValueError, match="[Rr]e-select"):
            get_required_select(None, 'Model')


class TestWrongTypes:
    def test_list_is_rejected(self):
        with pytest.raises(ValueError, match="Model"):
            get_required_select(['sam_vit_h_4b8939'], 'Model')

    def test_dict_is_rejected(self):
        with pytest.raises(ValueError, match="Model"):
            get_required_select({'0': True}, 'Model')

    def test_number_is_rejected(self):
        with pytest.raises(ValueError, match="Model"):
            get_required_select(3, 'Model')


class TestStaleValues:
    """A saved config can name an option that no longer exists — e.g. a SAM2
    checkpoint that was removed or renamed in a newer worker image."""

    def test_unknown_value_is_rejected_when_allowed_values_given(self):
        with pytest.raises(ValueError, match="old_model.pt"):
            get_required_select(
                'old_model.pt', 'Model', allowed_values=['new_model.pt'])

    def test_error_lists_available_options(self):
        with pytest.raises(ValueError, match="new_model.pt"):
            get_required_select(
                'old_model.pt', 'Model', allowed_values=['new_model.pt'])

    def test_no_allowed_values_means_any_string(self):
        # Workers whose models come from Girder (cellpose custom models) have
        # no static option list; only the shape is validated.
        assert get_required_select('my_custom_model', 'Model') == 'my_custom_model'
