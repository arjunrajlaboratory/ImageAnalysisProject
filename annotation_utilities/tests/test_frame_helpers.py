import sys
from pathlib import Path

import pytest


package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_root))

from annotation_utilities.annotation_tools import (
    frame_to_large_image_params,
    get_frame_index,
)


# A frame from a real single-channel, 20-timepoint dataset. Note that there is no
# 'IndexC' key at all: Girder omits the index for any dimension of size one.
SINGLE_CHANNEL_FRAME = {'Channel': 'Default', 'Frame': 3, 'Index': 3, 'IndexT': 3}

FULL_FRAME = {'Channel': 'GFP', 'Frame': 7, 'Index': 7,
              'IndexXY': 1, 'IndexZ': 2, 'IndexT': 3, 'IndexC': 1}


def test_get_frame_index_reads_present_index():
    assert get_frame_index(FULL_FRAME, 'IndexC') == 1
    assert get_frame_index(FULL_FRAME, 'IndexT') == 3


def test_get_frame_index_defaults_missing_index_c_to_zero():
    """Single-channel datasets omit IndexC; the only valid channel is 0."""
    assert get_frame_index(SINGLE_CHANNEL_FRAME, 'IndexC') == 0


def test_get_frame_index_defaults_every_missing_dimension_to_zero():
    for dimension in ('IndexXY', 'IndexZ', 'IndexC'):
        assert get_frame_index(SINGLE_CHANNEL_FRAME, dimension) == 0


def test_get_frame_index_accepts_short_dimension_names():
    assert get_frame_index(FULL_FRAME, 'C') == 1
    assert get_frame_index(SINGLE_CHANNEL_FRAME, 'C') == 0
    assert get_frame_index(FULL_FRAME, 'XY') == 1


def test_get_frame_index_honors_explicit_default():
    assert get_frame_index(SINGLE_CHANNEL_FRAME, 'IndexC', default=-1) == -1


def test_get_frame_index_rejects_unknown_dimension():
    """A typo must fail loudly rather than silently reporting coordinate 0."""
    with pytest.raises(ValueError):
        get_frame_index(FULL_FRAME, 'Channel')
    with pytest.raises(ValueError):
        get_frame_index(FULL_FRAME, 'IndexQ')


def test_frame_to_large_image_params_maps_all_indices():
    assert frame_to_large_image_params(FULL_FRAME) == {
        'xy': 1, 'z': 2, 't': 3, 'c': 1}


def test_frame_to_large_image_params_omits_absent_dimensions():
    """Dimensions the dataset does not use must not be passed to addTile."""
    assert frame_to_large_image_params(SINGLE_CHANNEL_FRAME) == {'t': 3}


def test_frame_to_large_image_params_skips_bare_index_key():
    """'Index' is the flat frame number, not an addTile parameter."""
    params = frame_to_large_image_params({'Index': 5, 'IndexZ': 2})
    assert params == {'z': 2}


def test_frame_to_large_image_params_ignores_non_index_metadata():
    params = frame_to_large_image_params(
        {'Channel': 'DAPI', 'Frame': 0, 'Index': 0, 'IndexC': 0})
    assert params == {'c': 0}


def test_frame_to_large_image_params_passes_through_unusual_axes():
    """An unrecognized axis keeps its own plane rather than being silently dropped."""
    assert frame_to_large_image_params({'IndexQ': 4}) == {'q': 4}


def test_frame_to_large_image_params_on_frame_without_indices():
    assert frame_to_large_image_params({'Channel': 'Default', 'Index': 0}) == {}
