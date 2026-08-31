from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))


try:
    import annotation_client.workers  # noqa: F401
except (ImportError, ModuleNotFoundError):
    annotation_client = types.ModuleType("annotation_client")
    annotation_client.__path__ = []
    workers_module = types.ModuleType("annotation_client.workers")
    utils_module = types.ModuleType("annotation_client.utils")
    workers_module.UPennContrastWorkerPreviewClient = MagicMock()
    utils_module.sendError = MagicMock()
    utils_module.sendProgress = MagicMock()
    utils_module.sendWarning = MagicMock()
    sys.modules.update(
        {
            "annotation_client": annotation_client,
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

    def get_required_select(value, field_name, allowed_values=None):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"The '{field_name}' setting has no valid value")
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(f"The '{field_name}' setting is stale: {value!r}")
        return value

    annotation_tools.get_required_select = get_required_select
    sys.modules.update(
        {
            "annotation_utilities": annotation_utilities,
            "annotation_utilities.annotation_tools": annotation_tools,
        }
    )

from illumination import IlluminationModel  # noqa: E402
from pipeline import (  # noqa: E402
    DownloadedSources,
    RawCompositeRequiredError,
    RawTrainingData,
    SourceLayout,
)
from refinement import PairMeasurement, RefinementResult  # noqa: E402
from entrypoint import ALGORITHM_OPTIONS, compute, interface  # noqa: E402


def _params():
    return {
        "workerInterface": {
            "Refine stitch positions": True,
            "Refinement channel": 0,
            "NCC threshold": 0.5,
            "Illumination algorithm": ALGORITHM_OPTIONS[0],
            "Output filename": "",
        }
    }


def test_interface_surfaces_position_and_annotation_warning() -> None:
    with patch("entrypoint.workers.UPennContrastWorkerPreviewClient") as preview:
        interface("image", "http://api", "token")

    values = preview.return_value.setWorkerImageInterface.call_args.args[1]
    assert values["Refine stitch positions"]["default"] is True
    assert values["Refinement channel"]["type"] == "channel"
    assert values["NCC threshold"]["default"] == 0.5
    ncc_tooltip = values["NCC threshold"]["tooltip"]
    assert "Lower values keep more" in ncc_tooltip
    assert "Higher values" in ncc_tooltip
    assert "disconnect" in ncc_tooltip
    assert values["Illumination algorithm"]["items"] == list(ALGORITHM_OPTIONS)
    assert "annotations are not moved" in values["Correction details"]["value"]
    assert "already-stitched TIFF-only" in values["Correction details"]["value"]
    assert sorted(field["displayOrder"] for field in values.values()) == list(
        range(len(values))
    )


def test_compute_refines_fits_all_channels_converts_and_uploads() -> None:
    girder = MagicMock()
    downloaded = DownloadedSources(
        document={"sources": []},
        document_path=Path("multi-source2.json"),
        document_item_id="multi-item",
        source_path=Path("source.nd2"),
        source_item_id="source-item",
    )
    training = RawTrainingData(
        reference_tiles=(np.ones((8, 8)), np.ones((8, 8))),
        model_stacks=np.ones((2, 2, 8, 8), dtype=np.float32),
        stage_positions_um=np.asarray(((0.0, 0.0), (100.0, 0.0))),
        channel_names=("DAPI", "YFP"),
        raw_shape=(8, 8),
        positions=2,
        time_points=1,
        z_planes=1,
        channels=2,
        home_z=0,
    )
    layout = SourceLayout(
        source_path="source.nd2",
        positions=np.asarray(((100, 100), (50, 100)), dtype=np.float64),
        linear_transform=-np.eye(2),
        source_position_indices=(),
        frames_per_position=2,
    )
    measurement = PairMeasurement(
        first=0,
        second=1,
        axis="horizontal",
        predicted_shift_x=50,
        predicted_shift_y=0,
        shift_x=51,
        shift_y=0,
        ncc=0.9,
        accepted=True,
    )
    refinement = RefinementResult(
        # Deliberately move the outer bound beyond the documented 16 px target.
        # A valid refinement is uploaded with a warning rather than discarded.
        positions=np.asarray(((120, 100), (50, 100))),
        measurements=(measurement,),
        accepted_measurements=(measurement,),
        residuals=(0.0,),
        max_residual=0.0,
        similarity_matrix=np.eye(2),
    )
    model = IlluminationModel(
        flatfield=np.ones((8, 8), dtype=np.float32),
        gains=np.ones(2, dtype=np.float32),
        diagnostics={"method": "overlap_dct_gain"},
    )
    corrected_document = {"sources": []}

    with (
        patch("entrypoint.GirderClient", return_value=girder),
        patch("entrypoint.download_composite_sources", return_value=downloaded),
        patch("entrypoint.ND2File") as nd2_file,
        patch("entrypoint.load_training_data", return_value=training),
        patch("entrypoint.parse_source_layout", return_value=layout),
        patch("entrypoint.refine_positions", return_value=refinement),
        patch("entrypoint.fit_overlap_dct", return_value=model) as fit,
        patch("entrypoint.write_corrected_tile_tiff"),
        patch(
            "entrypoint.corrected_source_document",
            return_value=corrected_document,
        ),
        patch("entrypoint.convert_multi_source"),
        patch("entrypoint.upload_result", return_value={"itemId": "output"}) as upload,
        patch("entrypoint.sendWarning") as send_warning,
    ):
        item = compute("dataset", "http://api", "token", _params())

    assert item == {"itemId": "output"}
    girder.setToken.assert_called_once_with("token")
    nd2_file.assert_called_once()
    assert fit.call_count == 2
    assert fit.call_args.kwargs["adaptive_tile_gains"] is True
    metadata = upload.call_args.args[-1]
    assert metadata["tool"] == "Stitch Refinement + Illumination Correction"
    assert metadata["worker_version"] == "1.0.1"
    assert metadata["refinement"]["pairs_matched"] == 1
    assert metadata["parameters"]["refinement_channel_name"] == "DAPI"
    assert metadata["source"]["original_nd2_item_id"] == "source-item"
    send_warning.assert_called_once()
    assert "coordinate-stability" in send_warning.call_args.args[0]


def test_compute_rejects_stale_algorithm_before_contacting_girder() -> None:
    params = _params()
    params["workerInterface"]["Illumination algorithm"] = "Removed method"

    with (
        patch("entrypoint.GirderClient") as girder,
        patch("entrypoint.sendError") as send_error,
    ):
        try:
            compute("dataset", "http://api", "token", params)
        except ValueError as exc:
            assert "Illumination algorithm" in str(exc)
        else:
            raise AssertionError("stale algorithm must fail")

    girder.assert_not_called()
    send_error.assert_called_once()


def test_compute_reports_that_stitched_only_input_is_unsupported() -> None:
    failure = RawCompositeRequiredError(
        "The dataset contains only an already-stitched TIFF-only image."
    )

    with (
        patch("entrypoint.GirderClient"),
        patch("entrypoint.download_composite_sources", side_effect=failure),
        patch("entrypoint.ND2File") as nd2_file,
        patch("entrypoint.sendError") as send_error,
    ):
        try:
            compute("dataset", "http://api", "token", _params())
        except RawCompositeRequiredError as exc:
            assert exc is failure
        else:
            raise AssertionError("stitched-only input must fail the job")

    send_error.assert_called_once_with(
        "Raw composite input required.", info=str(failure)
    )
    nd2_file.assert_not_called()
