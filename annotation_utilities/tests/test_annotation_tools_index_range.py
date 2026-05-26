import sys
from pathlib import Path

import numpy as np


package_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(package_root))

from annotation_utilities.annotation_tools import get_images_for_all_channels


class DummyTileClient:
    def __init__(self, tiles):
        self.tiles = tiles
        self.frame_requests = []

    def coordinatesToFrameIndex(self, xy, z, time, channel):
        self.frame_requests.append((xy, z, time, channel))
        return len(self.frame_requests) - 1

    def getRegion(self, dataset_id, frame):
        return np.full((2, 2), frame, dtype=np.uint8)


def test_get_images_for_all_channels_defaults_missing_index_range_to_one_channel():
    tile_client = DummyTileClient(tiles={})

    images = get_images_for_all_channels(tile_client, "dataset", 0, 0, 0)

    assert len(images) == 1
    assert tile_client.frame_requests == [(0, 0, 0, 0)]


def test_get_images_for_all_channels_defaults_missing_index_c_to_one_channel():
    tile_client = DummyTileClient(tiles={"IndexRange": {"IndexZ": 3}})

    images = get_images_for_all_channels(tile_client, "dataset", 0, 0, 0)

    assert len(images) == 1
    assert tile_client.frame_requests == [(0, 0, 0, 0)]


def test_get_images_for_all_channels_uses_index_c_when_present():
    tile_client = DummyTileClient(tiles={"IndexRange": {"IndexC": 3}})

    images = get_images_for_all_channels(tile_client, "dataset", 0, 0, 0)

    assert len(images) == 3
    assert tile_client.frame_requests == [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 0, 2),
    ]
