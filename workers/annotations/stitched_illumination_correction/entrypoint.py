import argparse
import importlib.metadata
import json
import sys

import numpy as np

import annotation_client.tiles as tiles
import annotation_client.workers as workers
from annotation_client.utils import sendError, sendProgress, sendWarning
import annotation_utilities.annotation_tools as annotation_tools

import illumination as correction


OUTPUT_PATH = "/tmp/illumination_corrected.tiff"
WORKER_NAME = "Stitched TIFF Illumination Correction"
WORKER_VERSION = "1.0.2"
ALGORITHM_OPTIONS = correction.ALGORITHM_OPTIONS
REFERENCE_CHANNEL_MODE_OPTIONS = correction.REFERENCE_CHANNEL_MODE_OPTIONS
DARKFIELD_OPTIONS = correction.DARKFIELD_OPTIONS
OUTPUT_TYPE_OPTIONS = correction.OUTPUT_TYPE_OPTIONS


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    values = {
        "Stitched TIFF illumination correction": {
            "type": "notes",
            "value": (
                "Stitched-TIFF fallback for grid-locked uneven illumination. "
                "Automatic mode fits on the reference Z plane, selects on held-out "
                "Z planes, and can leave a channel unchanged."
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
                "log-gradient, split-half affine, and no correction. It requires "
                "an independent Z plane for model selection."
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
            "tooltip": (
                "Physical grid and channel models are fitted at this XY. Only this "
                "XY and the reference time point are corrected."
            ),
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
            "tooltip": (
                "Time point used to fit the illumination models. Only this time "
                "point and the reference XY are corrected."
            ),
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
            "default": False,
            "tooltip": (
                "Experimental for BaSiC and folded log-gradient only. It estimates "
                "whole-tile gains from the fit plane and can remove real biology."
            ),
            "displayOrder": 9,
        },
        "Punctate channels for spot metric": {
            "type": "channelCheckboxes",
            "required": False,
            "tooltip": (
                "Use position-dependent spot counts as a soft selection metric only "
                "for channels known to contain punctate signal."
            ),
            "displayOrder": 10,
        },
        "Output type": {
            "type": "select",
            "items": list(OUTPUT_TYPE_OPTIONS),
            "default": "Float32 (recommended)",
            "tooltip": (
                "Float32 preserves the fitted numeric range for audit. Preserve "
                "source dtype rejects corrections that would materially clip."
            ),
            "displayOrder": 11,
        },
        "Validate every corrected plane": {
            "type": "checkbox",
            "default": True,
            "tooltip": (
                "Check object ranking, fine detail, and numeric range on every "
                "corrected frame before uploading."
            ),
            "displayOrder": 12,
        },
        "Minimum tile pitch": {
            "type": "number",
            "min": 16,
            "max": 5000,
            "default": 150,
            "unit": "pixels",
            "tooltip": "Smallest physical stitched-tile pitch to consider.",
            "displayOrder": 13,
        },
        "Maximum tile pitch": {
            "type": "number",
            "min": 32,
            "max": 10000,
            "default": 1400,
            "unit": "pixels",
            "tooltip": "Largest physical stitched-tile pitch to consider.",
            "displayOrder": 14,
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


def _dimension_size(tile_metadata, dimension):
    key = dimension if dimension.startswith("Index") else f"Index{dimension}"
    index_range = tile_metadata.get("IndexRange") or {}
    if key in index_range:
        return max(int(index_range[key]), 1)
    frames = tile_metadata.get("frames") or []
    if frames:
        return max(
            int(annotation_tools.get_frame_index(frame, key)) for frame in frames
        ) + 1
    if key == "IndexC" and tile_metadata.get("channels"):
        return max(len(tile_metadata["channels"]), 1)
    return 1


def _validate_reference_coordinates(coordinates, tile_metadata):
    dimensions = {"XY": "IndexXY", "Z": "IndexZ", "Time": "IndexT"}
    for label, dimension in dimensions.items():
        size = _dimension_size(tile_metadata, dimension)
        coordinate = coordinates[label]
        if coordinate >= size:
            raise ValueError(
                f"Reference {label} is {coordinate + 1}, but the dataset only has "
                f"{size} position{'s' if size != 1 else ''}"
            )


def _representative_validation_z(reference_z, num_z):
    candidates = {0, num_z // 2, num_z - 1}
    return sorted(z for z in candidates if 0 <= z < num_z and z != reference_z)


def _validate_output_values(image, output_type, source_dtype, tolerance=1e-4):
    values = np.asarray(image)
    if not np.isfinite(values).all():
        raise ValueError("The corrected output contains non-finite values")
    if output_type == OUTPUT_TYPE_OPTIONS[0]:
        return

    dtype = np.dtype(source_dtype)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
    elif np.issubdtype(dtype, np.floating):
        limits = np.finfo(dtype)
    else:
        return
    outside = (values < limits.min) | (values > limits.max)
    fraction = float(np.count_nonzero(outside) / values.size)
    if fraction > tolerance:
        raise ValueError(
            f"{fraction:.3%} of corrected pixels would be clipped when preserving "
            f"source dtype {dtype}; use Float32 output instead"
        )


def _cast_corrected(image, output_type, source_dtype):
    corrected = np.asarray(image, dtype=np.float32)
    if output_type == "Float32 (recommended)":
        return corrected
    dtype = np.dtype(source_dtype)
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        corrected = np.clip(np.rint(corrected), limits.min, limits.max)
    elif np.issubdtype(dtype, np.floating):
        limits = np.finfo(dtype)
        corrected = np.clip(corrected, limits.min, limits.max)
    return corrected.astype(dtype)


def _software_versions():
    versions = {"worker": WORKER_VERSION}
    for package in ("numpy", "scipy", "torch", "basicpy", "large-image"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


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


def _copy_sink_metadata(sink, tile_metadata):
    if "channels" in tile_metadata:
        sink.channelNames = tile_metadata["channels"]
    for name in ("mm_x", "mm_y", "magnification"):
        if name in tile_metadata:
            setattr(sink, name, tile_metadata[name])


def _phase_progress(start, stop):
    """Map a nested routine's local progress into one global worker phase."""
    start = float(start)
    stop = float(stop)

    def report(fraction, title, info):
        local = float(np.clip(fraction, 0.0, 1.0))
        sendProgress(start + (stop - start) * local, title, info)

    return report


def compute(datasetId, apiUrl, token, params):
    worker_interface = params.get("workerInterface", {})
    try:
        channels = annotation_tools.get_selected_channels(
            worker_interface.get("Channels to correct"), "Channels to correct"
        )
        punctate_channels = annotation_tools.get_selected_channels(
            worker_interface.get("Punctate channels for spot metric"),
            "Punctate channels for spot metric",
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
        per_tile_gain = worker_interface.get("Per-tile gain correction", False)
        validate_every_plane = worker_interface.get(
            "Validate every corrected plane", True
        )
        if not isinstance(per_tile_gain, bool):
            raise ValueError("Per-tile gain correction must be true or false")
        if not isinstance(validate_every_plane, bool):
            raise ValueError("Validate every corrected plane must be true or false")
        tile = params.get("tile", {})
        coordinates = {
            "XY": _parse_reference_coordinate(
                worker_interface.get("Reference XY"),
                tile.get("XY", 0),
                "Reference XY",
            ),
            "Z": _parse_reference_coordinate(
                worker_interface.get("Reference Z"),
                tile.get("Z", 0),
                "Reference Z",
            ),
            "Time": _parse_reference_coordinate(
                worker_interface.get("Reference Time"),
                tile.get("Time", 0),
                "Reference Time",
            ),
        }
        reference_channel_setting = 0
        if reference_channel_mode == REFERENCE_CHANNEL_MODE_OPTIONS[1]:
            reference_channel_setting = int(
                worker_interface.get("Reference channel")
            )
            if reference_channel_setting < 0:
                raise ValueError("Reference channel must be at least 0")
    except (TypeError, ValueError) as exc:
        sendError("Could not read the illumination settings.", info=str(exc))
        raise

    tile_client = tiles.UPennContrastDataset(
        apiUrl=apiUrl, token=token, datasetId=datasetId
    )
    try:
        _validate_reference_coordinates(coordinates, tile_client.tiles)
        num_channels = _dimension_size(tile_client.tiles, "IndexC")
        channels, missing_channels = annotation_tools.split_channel_selection(
            channels, num_channels
        )
        punctate_channels, missing_punctate = (
            annotation_tools.split_channel_selection(
                punctate_channels, num_channels
            )
        )
        if not channels:
            raise ValueError(
                "The selected channels do not exist in this dataset: "
                + ", ".join(str(channel + 1) for channel in missing_channels)
            )
        if missing_channels:
            sendWarning(
                "Some selected correction channels do not exist in this dataset.",
                info="Ignoring channels "
                + ", ".join(str(channel + 1) for channel in missing_channels),
            )
        if missing_punctate:
            sendWarning(
                "Some punctate-metric channels do not exist in this dataset.",
                info="Ignoring channels "
                + ", ".join(str(channel + 1) for channel in missing_punctate),
            )
        if (
            reference_channel_mode == REFERENCE_CHANNEL_MODE_OPTIONS[1]
            and reference_channel_setting >= num_channels
        ):
            raise ValueError(
                f"Reference channel {reference_channel_setting + 1} does not exist; "
                f"the dataset has {num_channels} channel"
                f"{'s' if num_channels != 1 else ''}"
            )
    except (TypeError, ValueError) as exc:
        sendError("Could not use the saved settings with this dataset.", info=str(exc))
        raise

    try:
        grid, reference_channel, reference_reports = correction.choose_reference_grid(
            tile_client,
            datasetId,
            coordinates,
            reference_channel_mode,
            reference_channel_setting,
            pitch_min,
            pitch_max,
            progress=_phase_progress(0.0, 0.15),
        )
    except Exception as exc:
        sendError("Could not determine the physical tile grid.", info=str(exc))
        raise

    validation_zs = _representative_validation_z(
        coordinates["Z"], _dimension_size(tile_client.tiles, "IndexZ")
    )
    selections = {}
    model_metadata = {}

    for index, channel in enumerate(channels):
        phase_start = 0.15 + 0.35 * index / max(len(channels), 1)
        phase_stop = 0.15 + 0.35 * (index + 1) / max(len(channels), 1)
        sendProgress(
            phase_start,
            WORKER_NAME,
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

            def validation_source(channel=channel):
                for z in validation_zs:
                    validation_frame = tile_client.coordinatesToFrameIndex(
                        coordinates["XY"], z, coordinates["Time"], channel
                    )
                    validation_plane = correction.as_plane(
                        tile_client.getRegion(datasetId, frame=validation_frame)
                    )
                    if validation_plane.shape != grid.shape:
                        raise ValueError(
                            f"Held-out Z {z + 1} for channel {channel + 1} has "
                            f"shape {validation_plane.shape}, but the shared grid "
                            f"expects {grid.shape}"
                        )
                    yield f"held-out Z {z + 1}", validation_plane

            selection = correction.select_model(
                raw,
                grid,
                algorithm,
                darkfield_mode,
                per_tile_gain,
                progress=_phase_progress(phase_start, phase_stop),
                validation_source=validation_source if validation_zs else None,
                use_spot_uniformity=channel in punctate_channels,
            )
        except Exception as exc:
            sendError(
                f"Could not fit a safe correction for channel {channel + 1}.",
                info=str(exc),
            )
            raise
        if selection.name == "identity":
            sendWarning(
                f"Channel {channel + 1} was left unchanged.",
                info=getattr(selection.model, "diagnostics", {}).get(
                    "reason",
                    "No correction beat the identity baseline by a reliable margin.",
                ),
            )
        if selection.candidate_failures:
            sendWarning(
                f"Correction candidate comparison was incomplete for channel "
                f"{channel + 1}.",
                info="; ".join(
                    f"{failure['name']} ({failure['kind']}): {failure['error']}"
                    for failure in selection.candidate_failures
                ),
            )
        selections[channel] = selection
        model_metadata[str(channel)] = {
            "channel_index_zero_based": channel,
            "selected": selection.name,
            "artifact_index": selection.artifact_index,
            "metrics": selection.metrics,
            "diagnostics": getattr(selection.model, "diagnostics", {}),
            "candidates": selection.alternatives,
            "candidate_failures": selection.candidate_failures,
        }

    # Delay the heavy writer import until validation and model fitting have passed.
    import large_image as li

    sink = li.new()
    frames = tile_client.tiles.get("frames")
    frame_records = list(enumerate(frames)) if frames else [(None, None)]

    for output_index, (frame_index, frame_metadata) in enumerate(frame_records):
        if frame_metadata is None:
            source_frame = tile_client.coordinatesToFrameIndex(
                coordinates["XY"],
                coordinates["Z"],
                coordinates["Time"],
                params.get("channel", 0),
            )
            channel = int(params.get("channel", 0))
            frame_xy = coordinates["XY"]
            frame_time = coordinates["Time"]
        else:
            source_frame = frame_index
            channel = int(annotation_tools.get_frame_index(frame_metadata, "IndexC"))
            frame_xy = int(
                annotation_tools.get_frame_index(frame_metadata, "IndexXY")
            )
            frame_time = int(
                annotation_tools.get_frame_index(frame_metadata, "IndexT")
            )
        raw = correction.as_plane(tile_client.getRegion(datasetId, frame=source_frame))
        source_dtype = _source_dtype(tile_client, raw.dtype)
        should_correct = (
            channel in selections
            and frame_xy == coordinates["XY"]
            and frame_time == coordinates["Time"]
        )

        if should_correct:
            try:
                corrected = selections[channel].model.apply(raw)
                _validate_output_values(corrected, output_type, source_dtype)
                output = _cast_corrected(corrected, output_type, source_dtype)
                _validate_output_values(output, output_type, source_dtype)
            except Exception as exc:
                sendError(
                    f"Could not create a safe output for frame {output_index + 1}.",
                    info=str(exc),
                )
                raise
            if validate_every_plane:
                validation = correction.preservation_metrics(raw, output)
                violations = validation["guardrail_violations"]
                if violations:
                    exc = ValueError("; ".join(violations))
                    sendError(
                        "Correction failed preservation checks on frame "
                        f"{output_index + 1}.",
                        info=str(exc),
                    )
                    raise exc
        elif output_type == OUTPUT_TYPE_OPTIONS[0]:
            output = raw.astype(np.float32)
        else:
            output = raw.astype(source_dtype, copy=False)

        if frame_metadata is None:
            sink.addTile(output, 0, 0, z=0)
        else:
            sink.addTile(
                output,
                0,
                0,
                **annotation_tools.frame_to_large_image_params(frame_metadata),
            )
        sendProgress(
            0.5 + 0.48 * (output_index + 1) / len(frame_records),
            WORKER_NAME,
            f"Processing frame {output_index + 1}/{len(frame_records)}",
        )

    _copy_sink_metadata(sink, tile_client.tiles)
    sink.write(OUTPUT_PATH)
    item = tile_client.client.uploadFileToFolder(datasetId, OUTPUT_PATH)
    metadata = _json_safe(
        {
            "tool": WORKER_NAME,
            "worker_version": WORKER_VERSION,
            "software_versions": _software_versions(),
            "input_representation": "stitched TIFF fallback",
            "indexing": "channel and coordinate keys marked zero_based are 0-based",
            "algorithm_requested": algorithm,
            "reference_channel_mode": reference_channel_mode,
            "reference_channel_zero_based": reference_channel,
            "reference_channel_one_based": reference_channel + 1,
            "reference_coordinates_zero_based": coordinates,
            "reference_coordinates_one_based": {
                key: value + 1 for key, value in coordinates.items()
            },
            "held_out_z_zero_based": validation_zs,
            "held_out_z_one_based": [z + 1 for z in validation_zs],
            "reference_candidates": reference_reports,
            "grid": grid.as_dict(),
            "per_tile_gain": per_tile_gain,
            "corrected_channels_zero_based": channels,
            "corrected_channels_one_based": [channel + 1 for channel in channels],
            "punctate_channels_zero_based": punctate_channels,
            "punctate_channels_one_based": [
                channel + 1 for channel in punctate_channels
            ],
            "output_type": output_type,
            "validated_every_plane": validate_every_plane,
            "correction_scope": {
                "XY_zero_based": coordinates["XY"],
                "Time_zero_based": coordinates["Time"],
                "Z": "all planes at the reference XY and time",
            },
            "pitch_bounds_pixels": {"minimum": pitch_min, "maximum": pitch_max},
            "darkfield_mode": darkfield_mode,
            "channel_models": model_metadata,
        }
    )
    tile_client.client.addMetadataToItem(item["itemId"], metadata)
    sendProgress(1.0, WORKER_NAME, "Corrected image uploaded")


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
