import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


# Native development runs do not install the NimbusImage client packages. The
# Docker test image does, so only provide lightweight import shims when absent.
try:
    import annotation_client.tiles  # noqa: F401
except (ImportError, ModuleNotFoundError):
    annotation_client = types.ModuleType("annotation_client")
    annotation_client.__path__ = []
    tiles_module = types.ModuleType("annotation_client.tiles")
    workers_module = types.ModuleType("annotation_client.workers")
    utils_module = types.ModuleType("annotation_client.utils")
    tiles_module.UPennContrastDataset = MagicMock()
    workers_module.UPennContrastWorkerPreviewClient = MagicMock()
    utils_module.sendError = MagicMock()
    utils_module.sendProgress = MagicMock()
    sys.modules.update(
        {
            "annotation_client": annotation_client,
            "annotation_client.tiles": tiles_module,
            "annotation_client.workers": workers_module,
            "annotation_client.utils": utils_module,
        }
    )

try:
    import annotation_utilities.annotation_tools  # noqa: F401
except (ImportError, ModuleNotFoundError):
    annotation_utilities = types.ModuleType("annotation_utilities")
    annotation_utilities.__path__ = []
    annotation_tools = types.ModuleType("annotation_utilities.annotation_tools")

    def get_selected_channels(value, field_name="channel selection"):
        if value in (None, "", {}):
            return []
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a mapping")
        return sorted(int(key) for key, selected in value.items() if selected)

    def get_required_select(value, field_name, allowed_values=None):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"The '{field_name}' setting has no valid value")
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(f"The '{field_name}' setting is stale: {value!r}")
        return value

    annotation_tools.get_selected_channels = get_selected_channels
    annotation_tools.get_required_select = get_required_select
    sys.modules.update(
        {
            "annotation_utilities": annotation_utilities,
            "annotation_utilities.annotation_tools": annotation_tools,
        }
    )

from illumination import CandidateResult, TileGrid  # noqa: E402
from entrypoint import compute, interface  # noqa: E402


def _grid(shape=(64, 64)):
    return TileGrid(
        pitch_y=16.0,
        pitch_x=16.0,
        seam_y=0.0,
        seam_x=0.0,
        height=shape[0],
        width=shape[1],
        seams_y=(0.0, 16.0, 32.0, 48.0, 64.0),
        seams_x=(0.0, 16.0, 32.0, 48.0, 64.0),
        seam_residual_y=0.0,
        seam_residual_x=0.0,
    )


class _AddOneModel:
    name = "fold_log_gradient"
    diagnostics = {"test": True}

    def apply(self, image):
        return np.asarray(image, dtype=np.float32) + 1.0


def _selection():
    return CandidateResult(
        name="fold_log_gradient",
        model=_AddOneModel(),
        metrics={
            "A1_fold_amp_rel_pct": 1.0,
            "P1_spot_uniformity": 1.0,
            "P2_spearman": 0.99,
            "P3_hf_power_ratio": 1.0,
            "P5_frac_nonpositive": 0.0,
        },
        artifact_index=0.25,
        violations=[],
        physics_violations=[],
        complexity=1,
    )


def _params(channels=None):
    return {
        "channel": 0,
        "tile": {"XY": 0, "Z": 0, "Time": 0},
        "workerInterface": {
            "Channels to correct": channels or {"0": True, "1": False},
            "Algorithm": "Automatic (recommended)",
            "Reference channel mode": "Automatically choose best channel",
            "Reference channel": 0,
            "Reference XY": "",
            "Reference Z": "",
            "Reference Time": "",
            "BaSiC darkfield": "Automatic",
            "Per-tile gain correction": True,
            "Output type": "Float32 (recommended)",
            "Validate every corrected plane": False,
            "Minimum tile pitch": 10,
            "Maximum tile pitch": 30,
        },
    }


def _tile_client(frames=True):
    client = MagicMock()
    client.tiles = {
        "IndexRange": {"IndexXY": 1, "IndexZ": 1, "IndexT": 1, "IndexC": 2},
        "channels": ["DAPI", "YFP"],
        "mm_x": 0.001,
        "mm_y": 0.001,
        "magnification": 20,
        "dtype": np.dtype("uint16"),
    }
    if frames:
        client.tiles["frames"] = [
            {"IndexXY": 0, "IndexZ": 0, "IndexT": 0, "IndexC": 0},
            {"IndexXY": 0, "IndexZ": 0, "IndexT": 0, "IndexC": 1},
        ]
    client.coordinatesToFrameIndex.side_effect = lambda xy, z, time, channel: channel
    client.getRegion.side_effect = lambda dataset_id, frame: np.full(
        (64, 64, 1), 100 + frame, dtype=np.uint16
    )
    client.client.uploadFileToFolder.return_value = {"itemId": "output-id"}
    return client


def test_interface_exposes_auto_and_manual_controls():
    with patch("entrypoint.workers.UPennContrastWorkerPreviewClient") as preview_cls:
        interface("image-id", "http://api", "token")

    values = preview_cls.return_value.setWorkerImageInterface.call_args.args[1]
    assert values["Illumination correction"]["type"] == "notes"
    assert values["Channels to correct"]["type"] == "channelCheckboxes"
    assert values["Algorithm"]["items"] == [
        "Automatic (recommended)",
        "BaSiC",
        "Folded log-gradient",
        "Split-half affine",
    ]
    assert values["Reference channel mode"]["type"] == "select"
    assert values["Reference channel"]["type"] == "channel"
    assert values["Output type"]["default"] == "Float32 (recommended)"
    assert sorted(v["displayOrder"] for v in values.values()) == list(
        range(len(values))
    )


def test_compute_corrects_selected_channels_and_uploads_tiff():
    tile_client = _tile_client()
    sink = MagicMock()
    fake_large_image = types.SimpleNamespace(new=MagicMock(return_value=sink))
    reference = (_grid(), 1, [{"channel": 1, "quality_score": 0.1}])

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid", return_value=reference),
        patch(
            "entrypoint.correction.select_model", return_value=_selection()
        ) as select_model,
        patch.dict(sys.modules, {"large_image": fake_large_image}),
    ):
        compute("dataset-id", "http://api", "token", _params())

    select_model.assert_called_once()
    assert sink.addTile.call_count == 2
    corrected = sink.addTile.call_args_list[0].args[0]
    untouched = sink.addTile.call_args_list[1].args[0]
    assert corrected.dtype == np.float32
    assert np.all(corrected == 101.0)
    assert untouched.dtype == np.float32
    assert np.all(untouched == 101)
    sink.write.assert_called_once_with("/tmp/illumination_corrected.tiff")
    tile_client.client.uploadFileToFolder.assert_called_once_with(
        "dataset-id", "/tmp/illumination_corrected.tiff"
    )
    metadata = tile_client.client.addMetadataToItem.call_args.args[1]
    assert metadata["tool"] == "Illumination correction"
    assert metadata["reference_channel"] == 1
    assert metadata["channel_models"]["0"]["selected"] == "fold_log_gradient"


def test_compute_rejects_malformed_channel_selection():
    tile_client = _tile_client()
    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.sendError") as send_error,
        pytest.raises(ValueError, match="mapping"),
    ):
        compute("dataset-id", "http://api", "token", _params(channels=[0]))

    send_error.assert_called_once()
    assert "channel selection" in send_error.call_args.args[0].lower()
    tile_client.client.uploadFileToFolder.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Algorithm", None),
        ("Algorithm", "Removed algorithm"),
        ("Reference channel mode", None),
        ("BaSiC darkfield", "Maybe"),
        ("Output type", None),
    ],
)
def test_compute_rejects_missing_or_stale_select_values(field, value):
    params = _params()
    params["workerInterface"][field] = value

    with (
        patch("entrypoint.tiles.UPennContrastDataset") as dataset_client,
        patch("entrypoint.sendError") as send_error,
        pytest.raises(ValueError, match=field),
    ):
        compute("dataset-id", "http://api", "token", params)

    send_error.assert_called_once()
    dataset_client.assert_not_called()


def test_compute_supports_single_frame_without_index_range():
    tile_client = _tile_client(frames=False)
    tile_client.tiles.pop("IndexRange")
    tile_client.tiles["channels"] = ["DAPI"]
    sink = MagicMock()
    fake_large_image = types.SimpleNamespace(new=MagicMock(return_value=sink))
    reference = (_grid(), 0, [{"channel": 0, "quality_score": 0.1}])
    params = _params(channels={"0": True})

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid", return_value=reference),
        patch("entrypoint.correction.select_model", return_value=_selection()),
        patch.dict(sys.modules, {"large_image": fake_large_image}),
    ):
        compute("dataset-id", "http://api", "token", params)

    sink.addTile.assert_called_once()
    tile_client.client.uploadFileToFolder.assert_called_once()
