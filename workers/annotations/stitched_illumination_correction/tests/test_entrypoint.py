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
    utils_module.sendWarning = MagicMock()
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

    def split_channel_selection(selected_channels, num_channels):
        present = sorted(
            {value for value in selected_channels if 0 <= value < num_channels}
        )
        missing = sorted({value for value in selected_channels if value not in present})
        return present, missing

    def get_frame_index(frame, dimension, default=0):
        key = dimension if dimension.startswith("Index") else f"Index{dimension}"
        return frame.get(key, default)

    def frame_to_large_image_params(frame):
        return {
            key.lower()[5:]: value
            for key, value in frame.items()
            if key.startswith("Index") and len(key) > 5
        }

    annotation_tools.get_selected_channels = get_selected_channels
    annotation_tools.get_required_select = get_required_select
    annotation_tools.split_channel_selection = split_channel_selection
    annotation_tools.get_frame_index = get_frame_index
    annotation_tools.frame_to_large_image_params = frame_to_large_image_params
    sys.modules.update(
        {
            "annotation_utilities": annotation_utilities,
            "annotation_utilities.annotation_tools": annotation_tools,
        }
    )

from illumination import CandidateResult, TileGrid  # noqa: E402
from entrypoint import WORKER_NAME, WORKER_VERSION, compute, interface  # noqa: E402


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
            "Punctate channels for spot metric": {},
            "Algorithm": "Automatic (recommended)",
            "Reference channel mode": "Automatically choose best channel",
            "Reference channel": 0,
            "Reference XY": "",
            "Reference Z": "",
            "Reference Time": "",
            "BaSiC darkfield": "Automatic",
            "Per-tile gain correction": False,
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
    assert WORKER_NAME == "Stitched TIFF Illumination Correction"
    assert WORKER_VERSION == "1.0.3"
    assert values["Stitched TIFF illumination correction"]["type"] == "notes"
    assert values["Channels to correct"]["type"] == "channelCheckboxes"
    assert values["Algorithm"]["items"] == [
        "Automatic (recommended)",
        "BaSiC",
        "Folded log-gradient",
        "Split-half affine",
    ]
    assert values["Reference channel mode"]["type"] == "select"
    assert values["Reference channel"]["type"] == "channel"
    assert values["Punctate channels for spot metric"]["type"] == "channelCheckboxes"
    assert values["Per-tile gain correction"]["default"] is False
    assert values["Output type"]["default"] == "Float32 (recommended)"
    assert sorted(v["displayOrder"] for v in values.values()) == list(
        range(len(values))
    )


def test_docker_identity_is_distinct_and_cpu_routed():
    dockerfile = (WORKER_DIR / "Dockerfile").read_text()

    assert 'isGPUWorker="false"' in dockerfile
    assert 'workerVersion="1.0.3"' in dockerfile
    assert 'interfaceName="Stitched TIFF Illumination Correction"' in dockerfile


def test_production_pyvips_runtime_can_initialize():
    """The TIFF sink depends on pyvips loading, not merely being installed."""
    import pyvips

    assert pyvips.API_mode
    assert tuple(pyvips.version(index) for index in range(3)) == (8, 18, 2)


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
    assert metadata["tool"] == WORKER_NAME
    assert metadata["worker_version"] == WORKER_VERSION
    assert metadata["reference_channel_zero_based"] == 1
    assert metadata["reference_channel_one_based"] == 2
    assert metadata["reference_coordinates_zero_based"] == {
        "XY": 0,
        "Z": 0,
        "Time": 0,
    }
    assert metadata["correction_scope"]["Z"].startswith("all planes")
    assert metadata["channel_models"]["0"]["selected"] == "fold_log_gradient"


def test_compute_supplies_independent_z_planes_for_automatic_selection():
    tile_client = _tile_client()
    tile_client.tiles["IndexRange"]["IndexZ"] = 2
    tile_client.coordinatesToFrameIndex.side_effect = (
        lambda xy, z, time, channel: z * 2 + channel
    )
    sink = MagicMock()
    fake_large_image = types.SimpleNamespace(new=MagicMock(return_value=sink))
    reference = (_grid(), 0, [{"channel": 0, "quality_score": 0.1}])

    def inspect_validation(*args, validation_source=None, **kwargs):
        reports = list(validation_source())
        assert [label for label, _ in reports] == ["held-out Z 2"]
        assert np.all(reports[0][1] == 102)
        return _selection()

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid", return_value=reference),
        patch("entrypoint.correction.select_model", side_effect=inspect_validation),
        patch.dict(sys.modules, {"large_image": fake_large_image}),
    ):
        compute("dataset-id", "http://api", "token", _params())

    tile_client.client.uploadFileToFolder.assert_called_once()


def test_compute_warns_when_automatic_algorithm_panel_is_incomplete():
    tile_client = _tile_client()
    sink = MagicMock()
    fake_large_image = types.SimpleNamespace(new=MagicMock(return_value=sink))
    reference = (_grid(), 0, [{"channel": 0, "quality_score": 0.1}])
    selection = _selection()
    selection.candidate_failures = [
        {
            "name": "basic_darkfield_on",
            "kind": "unavailable",
            "error": "not applicable to this image",
        }
    ]

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid", return_value=reference),
        patch("entrypoint.correction.select_model", return_value=selection),
        patch("entrypoint.sendWarning") as send_warning,
        patch.dict(sys.modules, {"large_image": fake_large_image}),
    ):
        compute("dataset-id", "http://api", "token", _params())

    assert any(
        "incomplete" in call.args[0].lower()
        for call in send_warning.call_args_list
    )
    metadata = tile_client.client.addMetadataToItem.call_args.args[1]
    assert metadata["channel_models"]["0"]["candidate_failures"] == (
        selection.candidate_failures
    )


def test_compute_reports_monotonic_global_progress():
    tile_client = _tile_client()
    sink = MagicMock()
    fake_large_image = types.SimpleNamespace(new=MagicMock(return_value=sink))
    reference = (_grid(), 0, [{"channel": 0, "quality_score": 0.1}])
    progress_values = []

    def choose_grid(*args, progress=None, **kwargs):
        progress(0.0, "grid", "start")
        progress(0.8, "grid", "nearly done")
        return reference

    def select_model(*args, progress=None, **kwargs):
        progress(0.0, "model", "start")
        progress(0.75, "model", "nearly done")
        return _selection()

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid", side_effect=choose_grid),
        patch("entrypoint.correction.select_model", side_effect=select_model),
        patch(
            "entrypoint.sendProgress",
            side_effect=lambda fraction, *args: progress_values.append(fraction),
        ),
        patch.dict(sys.modules, {"large_image": fake_large_image}),
    ):
        compute("dataset-id", "http://api", "token", _params())

    assert progress_values == sorted(progress_values)
    assert progress_values[-1] == 1.0


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


def test_compute_rejects_selected_channels_outside_dataset():
    tile_client = _tile_client()
    params = _params(channels={"3": True})

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid") as choose_grid,
        patch("entrypoint.sendError") as send_error,
        pytest.raises(ValueError, match="do not exist"),
    ):
        compute("dataset-id", "http://api", "token", params)

    choose_grid.assert_not_called()
    send_error.assert_called_once()


def test_compute_rejects_reference_coordinate_outside_dataset():
    tile_client = _tile_client()
    params = _params()
    params["workerInterface"]["Reference Z"] = "2"

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid") as choose_grid,
        patch("entrypoint.sendError") as send_error,
        pytest.raises(ValueError, match="Reference Z"),
    ):
        compute("dataset-id", "http://api", "token", params)

    choose_grid.assert_not_called()
    send_error.assert_called_once()


def test_automatic_reference_ignores_malformed_manual_channel_value():
    tile_client = _tile_client()
    sink = MagicMock()
    fake_large_image = types.SimpleNamespace(new=MagicMock(return_value=sink))
    reference = (_grid(), 0, [{"channel": 0, "quality_score": 0.1}])
    params = _params()
    params["workerInterface"]["Reference channel"] = None

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid", return_value=reference),
        patch("entrypoint.correction.select_model", return_value=_selection()),
        patch.dict(sys.modules, {"large_image": fake_large_image}),
    ):
        compute("dataset-id", "http://api", "token", params)

    tile_client.client.uploadFileToFolder.assert_called_once()


def test_compute_only_applies_model_to_reference_xy_and_time():
    tile_client = _tile_client()
    tile_client.tiles["IndexRange"]["IndexXY"] = 2
    tile_client.tiles["frames"] = [
        {"IndexXY": 0, "IndexZ": 0, "IndexT": 0, "IndexC": 0},
        {"IndexXY": 1, "IndexZ": 0, "IndexT": 0, "IndexC": 0},
    ]
    tile_client.getRegion.side_effect = lambda dataset_id, frame: np.full(
        (64, 64, 1), 100 + frame, dtype=np.uint16
    )
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

    reference_output = sink.addTile.call_args_list[0].args[0]
    other_xy_output = sink.addTile.call_args_list[1].args[0]
    assert np.all(reference_output == 101)
    assert np.all(other_xy_output == 101)


class _OverflowModel:
    name = "fold_log_gradient"
    diagnostics = {}

    def apply(self, image):
        return np.full(np.asarray(image).squeeze().shape, 70000.0, dtype=np.float32)


def test_preserve_dtype_rejects_material_clipping():
    tile_client = _tile_client()
    sink = MagicMock()
    fake_large_image = types.SimpleNamespace(new=MagicMock(return_value=sink))
    reference = (_grid(), 0, [{"channel": 0, "quality_score": 0.1}])
    selection = _selection()
    selection.model = _OverflowModel()
    params = _params(channels={"0": True})
    params["workerInterface"]["Output type"] = "Preserve source dtype"

    with (
        patch("entrypoint.tiles.UPennContrastDataset", return_value=tile_client),
        patch("entrypoint.correction.choose_reference_grid", return_value=reference),
        patch("entrypoint.correction.select_model", return_value=selection),
        patch("entrypoint.sendError") as send_error,
        patch.dict(sys.modules, {"large_image": fake_large_image}),
        pytest.raises(ValueError, match="clipped"),
    ):
        compute("dataset-id", "http://api", "token", params)

    send_error.assert_called()
    sink.write.assert_not_called()
