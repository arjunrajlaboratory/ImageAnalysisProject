from unittest.mock import MagicMock, patch
import sys
import types

import numpy as np

annotation_client = types.ModuleType("annotation_client")
annotation_client.workers = types.ModuleType("annotation_client.workers")
annotation_client.annotations = types.ModuleType("annotation_client.annotations")
annotation_client.tiles = types.ModuleType("annotation_client.tiles")
annotation_client.utils = types.ModuleType("annotation_client.utils")
annotation_client.workers.UPennContrastWorkerPreviewClient = object
annotation_client.workers.UPennContrastWorkerClient = object
annotation_client.annotations.UPennContrastAnnotationClient = object
annotation_client.tiles.UPennContrastDataset = object
annotation_client.utils.sendProgress = lambda *args, **kwargs: None

sys.modules.setdefault("annotation_client", annotation_client)
sys.modules.setdefault("annotation_client.workers", annotation_client.workers)
sys.modules.setdefault("annotation_client.annotations", annotation_client.annotations)
sys.modules.setdefault("annotation_client.tiles", annotation_client.tiles)
sys.modules.setdefault("annotation_client.utils", annotation_client.utils)

from entrypoint import compute, interface


def _params(all_channels=True, selected_channel=0, file_name="line_scan.csv"):
    return {
        "workerInterface": {
            "All channels": all_channels,
            "Channel": selected_channel,
            "File name": file_name,
        },
    }


def _line_annotation():
    return {
        "_id": "line_1",
        "coordinates": [
            {"x": 1.5, "y": 1.5},
            {"x": 5.5, "y": 1.5},
        ],
        "location": {"Time": 0, "Z": 0, "XY": 0},
        "tags": ["scan"],
    }


def test_interface_registers_expected_fields():
    with patch("annotation_client.workers.UPennContrastWorkerPreviewClient") as preview_cls:
        interface("image", "http://api", "token")

    preview_cls.return_value.setWorkerImageInterface.assert_called_once()
    interface_data = preview_cls.return_value.setWorkerImageInterface.call_args[0][1]
    assert {"Line Scan CSV", "All channels", "Channel", "File name"} <= set(interface_data)


def test_all_channels_defaults_to_single_channel_without_index_range():
    with (
        patch("annotation_client.workers.UPennContrastWorkerClient") as worker_cls,
        patch("annotation_client.annotations.UPennContrastAnnotationClient") as annotation_cls,
        patch("annotation_client.tiles.UPennContrastDataset") as dataset_cls,
    ):
        worker_cls.return_value.get_annotation_list_by_shape.return_value = [_line_annotation()]
        annotation_cls.return_value.client = MagicMock()
        annotation_cls.return_value.client.getFolder.return_value = {"_id": "folder_id"}
        dataset_cls.return_value.tiles = {}
        dataset_cls.return_value.coordinatesToFrameIndex.side_effect = (
            lambda xy, z, time, channel: channel
        )
        dataset_cls.return_value.getRegion.return_value = np.full((10, 10), 42, dtype=np.uint8)

        compute("dataset", "http://api", "token", _params(all_channels=True))

    dataset_cls.return_value.coordinatesToFrameIndex.assert_called_once_with(0, 0, 0, 0)
    assert dataset_cls.return_value.getRegion.call_count == 1
    annotation_cls.return_value.client.uploadStreamToFolder.assert_called_once()


def test_all_channels_defaults_to_single_channel_without_index_c():
    with (
        patch("annotation_client.workers.UPennContrastWorkerClient") as worker_cls,
        patch("annotation_client.annotations.UPennContrastAnnotationClient") as annotation_cls,
        patch("annotation_client.tiles.UPennContrastDataset") as dataset_cls,
    ):
        worker_cls.return_value.get_annotation_list_by_shape.return_value = [_line_annotation()]
        annotation_cls.return_value.client = MagicMock()
        annotation_cls.return_value.client.getFolder.return_value = {"_id": "folder_id"}
        dataset_cls.return_value.tiles = {"IndexRange": {"IndexZ": 2}}
        dataset_cls.return_value.coordinatesToFrameIndex.side_effect = (
            lambda xy, z, time, channel: channel
        )
        dataset_cls.return_value.getRegion.return_value = np.full((10, 10), 7, dtype=np.uint8)

        compute("dataset", "http://api", "token", _params(all_channels=True))

    dataset_cls.return_value.coordinatesToFrameIndex.assert_called_once_with(0, 0, 0, 0)
    assert dataset_cls.return_value.getRegion.call_count == 1


def test_all_channels_uses_index_c_when_present():
    with (
        patch("annotation_client.workers.UPennContrastWorkerClient") as worker_cls,
        patch("annotation_client.annotations.UPennContrastAnnotationClient") as annotation_cls,
        patch("annotation_client.tiles.UPennContrastDataset") as dataset_cls,
    ):
        worker_cls.return_value.get_annotation_list_by_shape.return_value = [_line_annotation()]
        annotation_cls.return_value.client = MagicMock()
        annotation_cls.return_value.client.getFolder.return_value = {"_id": "folder_id"}
        dataset_cls.return_value.tiles = {"IndexRange": {"IndexC": 3}}
        dataset_cls.return_value.coordinatesToFrameIndex.side_effect = (
            lambda xy, z, time, channel: channel
        )
        dataset_cls.return_value.getRegion.return_value = np.full((10, 10), 7, dtype=np.uint8)

        compute("dataset", "http://api", "token", _params(all_channels=True))

    channels_loaded = [
        call.args[3] for call in dataset_cls.return_value.coordinatesToFrameIndex.call_args_list
    ]
    assert channels_loaded == [0, 1, 2]
    assert dataset_cls.return_value.getRegion.call_count == 3


def test_selected_channel_does_not_require_index_range():
    with (
        patch("annotation_client.workers.UPennContrastWorkerClient") as worker_cls,
        patch("annotation_client.annotations.UPennContrastAnnotationClient") as annotation_cls,
        patch("annotation_client.tiles.UPennContrastDataset") as dataset_cls,
    ):
        worker_cls.return_value.get_annotation_list_by_shape.return_value = [_line_annotation()]
        annotation_cls.return_value.client = MagicMock()
        annotation_cls.return_value.client.getFolder.return_value = {"_id": "folder_id"}
        dataset_cls.return_value.tiles = {}
        dataset_cls.return_value.coordinatesToFrameIndex.side_effect = (
            lambda xy, z, time, channel: channel
        )
        dataset_cls.return_value.getRegion.return_value = np.full((10, 10), 99, dtype=np.uint8)

        compute(
            "dataset",
            "http://api",
            "token",
            _params(all_channels=False, selected_channel=2),
        )

    dataset_cls.return_value.coordinatesToFrameIndex.assert_called_once_with(0, 0, 0, 2)
    assert dataset_cls.return_value.getRegion.call_count == 1
