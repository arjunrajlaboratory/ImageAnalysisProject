import argparse
import json
import sys

import numpy as np

import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendError, sendProgress
import annotation_utilities.annotation_tools as annotation_tools

import illumination as correction


OUTPUT_PATH = "/tmp/illumination_corrected.tiff"
ALGORITHM_OPTIONS = (
    "Automatic (recommended)",
    "BaSiC",
    "Folded log-gradient",
    "Split-half affine",
)
REFERENCE_CHANNEL_MODE_OPTIONS = (
    "Automatically choose best channel",
    "Use specified channel",
)
DARKFIELD_OPTIONS = ("Automatic", "Enabled", "Disabled")
OUTPUT_TYPE_OPTIONS = ("Float32 (recommended)", "Preserve source dtype")


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    values = {
        "Illumination correction": {
            "type": "notes",
            "value": (
                "Corrects grid-locked uneven illumination in stitched microscopy "
                "images. Automatic mode selects a shared physical tile grid, then "
                "fits and validates the best correction independently for each channel."
            ),
            "displayOrder": 0,
        },
        "Channels to correct": {
            "type": "channelCheckboxes",
            "required": True,
            "tooltip": "Channels to correct in the uploaded image.",
            "displayOrder": 1,
        },
        "Algorithm": {
            "type": "select",
            "items": list(ALGORITHM_OPTIONS),
            "default": "Automatic (recommended)",
            "tooltip": (
                "Automatic compares BaSiC with darkfield off/on, folded "
                "log-gradient, and split-half affine correction."
            ),
            "displayOrder": 2,
        },
        "Reference channel mode": {
            "type": "select",
            "items": list(REFERENCE_CHANNEL_MODE_OPTIONS),
            "default": "Automatically choose best channel",
            "tooltip": (
                "Automatic mode chooses the most reliable grid estimate in the "
                "dominant cross-channel pitch cluster."
            ),
            "displayOrder": 3,
        },
        "Reference channel": {
            "type": "channel",
            "default": 0,
            "tooltip": "Used only when Reference channel mode is manual.",
            "displayOrder": 4,
        },
        "Reference XY": {
            "type": "text",
            "default": "",
            "vueAttrs": {
                "placeholder": "blank = current XY",
                "label": "Reference XY (1-based)",
                "persistentPlaceholder": True,
                "filled": True,
            },
            "tooltip": "Physical grid and channel models are fitted at this XY.",
            "displayOrder": 5,
        },
        "Reference Z": {
            "type": "text",
            "default": "",
            "vueAttrs": {
                "placeholder": "blank = current Z",
                "label": "Reference Z (1-based)",
                "persistentPlaceholder": True,
                "filled": True,
            },
            "tooltip": "Use a well-focused plane; the fitted model is shared across Z.",
            "displayOrder": 6,
        },
        "Reference Time": {
            "type": "text",
            "default": "",
            "vueAttrs": {
                "placeholder": "blank = current time",
                "label": "Reference time (1-based)",
                "persistentPlaceholder": True,
                "filled": True,
            },
            "tooltip": "Time point used to fit the illumination models.",
            "displayOrder": 7,
        },
        "BaSiC darkfield": {
            "type": "select",
            "items": list(DARKFIELD_OPTIONS),
            "default": "Automatic",
            "tooltip": (
                "Automatic evaluates both settings. Darkfield is channel-specific "
                "and is rejected when it is physically implausible."
            ),
            "displayOrder": 8,
        },
        "Per-tile gain correction": {
            "type": "checkbox",
            "default": True,
            "tooltip": (
                "Correct residual whole-tile gain changes after fitting the shared "
                "within-tile field."
            ),
            "displayOrder": 9,
        },
        "Output type": {
            "type": "select",
            "items": list(OUTPUT_TYPE_OPTIONS),
            "default": "Float32 (recommended)",
            "tooltip": (
                "Float32 preserves the fitted numeric range for audit. Source dtype "
                "clips corrected values to the source range."
            ),
            "displayOrder": 10,
        },
        "Validate every corrected plane": {
            "type": "checkbox",
            "default": True,
            "tooltip": (
                "Check object ranking, fine detail, and numeric range on every "
                "corrected frame before uploading."
            ),
            "displayOrder": 11,
        },
        "Minimum tile pitch": {
            "type": "number",
            "min": 16,
            "max": 5000,
            "default": 150,
            "unit": "pixels",
            "tooltip": "Smallest physical stitched-tile pitch to consider.",
            "displayOrder": 12,
        },
        "Maximum tile pitch": {
            "type": "number",
            "min": 32,
            "max": 10000,
            "default": 1400,
            "unit": "pixels",
            "tooltip": "Largest physical stitched-tile pitch to consider.",
            "displayOrder": 13,
        },
    }
    client.setWorkerImageInterface(image, values)


def _parse_reference_coordinate(value, current, name):
    if value is None or str(value).strip() == "":
        return int(current)
    try:
        coordinate = int(str(value).strip()) - 1
    except ValueError as exc:
        raise ValueError(f"{name} must be a 1-based integer or left blank") from exc
    if coordinate < 0:
        raise ValueError(f"{name} must be at least 1")
    return coordinate


def _source_dtype(tile_client, fallback):
    try:
        return np.dtype(tile_client.tiles.get("dtype", fallback))
    except TypeError:
        return np.dtype(fallback)


def _cast_corrected(image, output_type, source_dtype):
    corrected = np.asarray(image, dtype=np.float32)
    if output_type == "Float32 (recommended)":
        return corrected
    dtype = np.dtype(source_dtype)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        corrected = np.clip(np.rint(corrected), limits.min, limits.max)
    return corrected.astype(dtype)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _frame_parameters(frame):
    return {
        key.lower()[5:]: value
        for key, value in frame.items()
        if key.startswith("Index") and len(key) > 5
    }


def _copy_sink_metadata(sink, tile_metadata):
    if "channels" in tile_metadata:
        sink.channelNames = tile_metadata["channels"]
    for name in ("mm_x", "mm_y", "magnification"):
        if name in tile_metadata:
            setattr(sink, name, tile_metadata[name])


def compute(datasetId, apiUrl, token, params):
    worker_interface = params.get("workerInterface", {})
    try:
        channels = annotation_tools.get_selected_channels(
            worker_interface.get("Channels to correct"), "Channels to correct"
        )
    except ValueError as exc:
        sendError("Could not read the channel selection.", info=str(exc))
        raise
    if not channels:
        exc = ValueError("Select at least one channel and run the worker again.")
        sendError(
            "No channels selected for illumination correction.",
            info=str(exc),
        )
        raise exc

    try:
        algorithm = annotation_tools.get_required_select(
            worker_interface.get("Algorithm"),
            "Algorithm",
            allowed_values=ALGORITHM_OPTIONS,
        )
        reference_channel_mode = annotation_tools.get_required_select(
            worker_interface.get("Reference channel mode"),
            "Reference channel mode",
            allowed_values=REFERENCE_CHANNEL_MODE_OPTIONS,
        )
        darkfield_mode = annotation_tools.get_required_select(
            worker_interface.get("BaSiC darkfield"),
            "BaSiC darkfield",
            allowed_values=DARKFIELD_OPTIONS,
        )
        output_type = annotation_tools.get_required_select(
            worker_interface.get("Output type"),
            "Output type",
            allowed_values=OUTPUT_TYPE_OPTIONS,
        )
        pitch_min = float(worker_interface.get("Minimum tile pitch", 150))
        pitch_max = float(worker_interface.get("Maximum tile pitch", 1400))
        if pitch_min <= 0 or pitch_max <= pitch_min:
            raise ValueError(
                "Maximum tile pitch must be greater than minimum tile pitch"
            )
        tile = params.get("tile", {})
        coordinates = {
            "XY": _parse_reference_coordinate(
                worker_interface.get("Reference XY"), tile.get("XY", 0), "Reference XY"
            ),
            "Z": _parse_reference_coordinate(
                worker_interface.get("Reference Z"), tile.get("Z", 0), "Reference Z"
            ),
            "Time": _parse_reference_coordinate(
                worker_interface.get("Reference Time"),
                tile.get("Time", 0),
                "Reference Time",
            ),
        }
        reference_channel_setting = int(worker_interface.get("Reference channel", 0))
    except (TypeError, ValueError) as exc:
        sendError("Could not read the illumination settings.", info=str(exc))
        raise

    tile_client = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId
    )
    try:
        grid, reference_channel, reference_reports = correction.choose_reference_grid(
            tile_client,
            datasetId,
            coordinates,
            reference_channel_mode,
            reference_channel_setting,
            pitch_min,
            pitch_max,
            progress=sendProgress,
        )
    except Exception as exc:
        sendError("Could not determine the physical tile grid.", info=str(exc))
        raise

    per_tile_gain = bool(worker_interface.get("Per-tile gain correction", True))
    selections = {}
    model_metadata = {}

    for index, channel in enumerate(channels):
        sendProgress(
            0.15 + 0.35 * index / max(len(channels), 1),
            "Illumination correction",
            f"Selecting a correction for channel {channel + 1}",
        )
        try:
            frame = tile_client.coordinatesToFrameIndex(
                coordinates["XY"], coordinates["Z"], coordinates["Time"], channel
            )
            raw = correction.as_plane(tile_client.getRegion(datasetId, frame=frame))
            if raw.shape != grid.shape:
                raise ValueError(
                    f"Reference channel {channel + 1} has shape {raw.shape}, but the "
                    f"shared grid expects {grid.shape}"
                )
            selection = correction.select_model(
                raw,
                grid,
                algorithm,
                darkfield_mode,
                per_tile_gain,
                progress=sendProgress,
            )
        except Exception as exc:
            sendError(
                f"Could not fit a safe correction for channel {channel + 1}.",
                info=str(exc),
            )
            raise
        selections[channel] = selection
        model_metadata[str(channel)] = {
            "selected": selection.name,
            "artifact_index": selection.artifact_index,
            "metrics": selection.metrics,
            "diagnostics": getattr(selection.model, "diagnostics", {}),
            "candidates": selection.alternatives,
        }

    # Delay the heavy writer import until validation and model fitting have passed.
    import large_image as li

    sink = li.new()
    frames = tile_client.tiles.get("frames")
    frame_records = list(enumerate(frames)) if frames else [(None, None)]
    validate_every_plane = bool(
        worker_interface.get("Validate every corrected plane", True)
    )

    for output_index, (frame_index, frame_metadata) in enumerate(frame_records):
        if frame_metadata is None:
            source_frame = tile_client.coordinatesToFrameIndex(
                coordinates["XY"],
                coordinates["Z"],
                coordinates["Time"],
                params.get("channel", 0),
            )
            channel = int(params.get("channel", 0))
        else:
            source_frame = frame_index
            channel = int(frame_metadata.get("IndexC", 0))
        raw = correction.as_plane(tile_client.getRegion(datasetId, frame=source_frame))
        source_dtype = _source_dtype(tile_client, raw.dtype)

        if channel in selections:
            try:
                corrected = selections[channel].model.apply(raw)
            except Exception as exc:
                sendError(
                    f"Could not apply the correction to frame {output_index + 1}.",
                    info=str(exc),
                )
                raise
            if validate_every_plane:
                validation = correction.preservation_metrics(raw, corrected)
                violations = validation["guardrail_violations"]
                if violations:
                    exc = ValueError("; ".join(violations))
                    sendError(
                        "Correction failed preservation checks on frame "
                        f"{output_index + 1}.",
                        info=str(exc),
                    )
                    raise exc
            output = _cast_corrected(corrected, output_type, source_dtype)
        elif output_type == "Float32 (recommended)":
            output = raw.astype(np.float32)
        else:
            output = raw.astype(source_dtype, copy=False)

        if frame_metadata is None:
            sink.addTile(output, 0, 0, z=0)
        else:
            sink.addTile(output, 0, 0, **_frame_parameters(frame_metadata))
        sendProgress(
            0.5 + 0.48 * (output_index + 1) / len(frame_records),
            "Illumination correction",
            f"Processing frame {output_index + 1}/{len(frame_records)}",
        )

    _copy_sink_metadata(sink, tile_client.tiles)
    sink.write(OUTPUT_PATH)
    item = tile_client.client.uploadFileToFolder(datasetId, OUTPUT_PATH)
    metadata = _json_safe(
        {
            "tool": "Illumination correction",
            "algorithm_requested": algorithm,
            "reference_channel_mode": reference_channel_mode,
            "reference_channel": reference_channel,
            "reference_coordinates": coordinates,
            "reference_candidates": reference_reports,
            "grid": grid.as_dict(),
            "per_tile_gain": per_tile_gain,
            "output_type": output_type,
            "validated_every_plane": validate_every_plane,
            "channel_models": model_metadata,
        }
    )
    tile_client.client.addMetadataToItem(item["itemId"], metadata)
    sendProgress(1.0, "Illumination correction", "Corrected image uploaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correct uneven image illumination")
    parser.add_argument("--datasetId", type=str, required=False, action="store")
    parser.add_argument("--apiUrl", type=str, required=True, action="store")
    parser.add_argument("--token", type=str, required=True, action="store")
    parser.add_argument("--request", type=str, required=True, action="store")
    parser.add_argument("--parameters", type=str, required=True, action="store")
    args = parser.parse_args(sys.argv[1:])

    parameters = json.loads(args.parameters)
    match args.request:
        case "compute":
            compute(args.datasetId, args.apiUrl, args.token, parameters)
        case "interface":
            interface(parameters["image"], args.apiUrl, args.token)
