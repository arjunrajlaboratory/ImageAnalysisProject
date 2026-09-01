import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from girder_client import GirderClient
from nd2 import ND2File

import annotation_client.workers as workers
from annotation_client.utils import sendError, sendProgress, sendWarning
import annotation_utilities.annotation_tools as annotation_tools

from illumination import fit_overlap_dct
from pipeline import (
    MODEL_SIZE,
    RawCompositeRequiredError,
    corrected_source_document,
    convert_multi_source,
    download_composite_sources,
    load_training_data,
    parse_source_layout,
    transformed_bounds,
    upload_result,
    write_corrected_tile_tiff,
)
from refinement import refine_positions


WORKER_NAME = "Stitch Refinement + Illumination Correction"
WORKER_VERSION = "1.0.2"
ALGORITHM_OPTIONS = (
    "Overlap DCT + tile gains (recommended)",
    "Overlap DCT",
)


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(apiUrl=apiUrl, token=token)
    values = {
        "Correction details": {
            "type": "notes",
            "value": (
                "Uses the original raw ND2 and deployed multi-source geometry to "
                "refine translations, correct raw-tile illumination, and upload a "
                "new pyramidal TIFF. Existing annotations are not moved; if this "
                "dataset already has annotations, their mosaic coordinates can shift "
                "by tens of pixels relative to the corrected image. The original "
                "image is not modified. Requires the original raw ND2 and "
                "multi-source2.json; already-stitched TIFF-only datasets cannot be "
                "used because their raw tile overlaps are unavailable."
            ),
            "displayOrder": 0,
        },
        "Refine stitch positions": {
            "type": "checkbox",
            "default": True,
            "tooltip": (
                "Adjust translations only using metadata-seeded adjacent-tile "
                "cross-correlation. Transform matrix coefficients are never changed."
            ),
            "displayOrder": 1,
        },
        "Refinement channel": {
            "type": "channel",
            "default": 0,
            "tooltip": (
                "Channel used for max-Z tile alignment. Channel 1 / DAPI is the "
                "validated default."
            ),
            "displayOrder": 2,
        },
        "NCC threshold": {
            "type": "number",
            "default": 0.5,
            "min": 0.5,
            "max": 1.0,
            "step": 0.05,
            "tooltip": (
                "Higher NCC means the overlap texture matches more reliably. The "
                "validated default is 0.5. Lower values keep more dim or low-texture "
                "pairs but can admit false matches. Higher values reject ambiguous "
                "pairs but can disconnect the tile grid or leave no usable pairs. "
                "Raise it only when low-NCC pairs report inconsistent shifts."
            ),
            "displayOrder": 3,
        },
        "Illumination algorithm": {
            "type": "select",
            "items": list(ALGORITHM_OPTIONS),
            "default": ALGORITHM_OPTIONS[0],
            "tooltip": (
                "The supplied raw-tile overlap-DCT model. The recommended option also "
                "fits small regularized per-tile gains."
            ),
            "displayOrder": 4,
        },
        "Output filename": {
            "type": "text",
            "default": "",
            "vueAttrs": {
                "label": "Optional output TIFF filename",
                "placeholder": "Automatically derived from the source ND2",
                "persistentPlaceholder": True,
                "filled": True,
            },
            "displayOrder": 5,
        },
    }
    client.setWorkerImageInterface(image, values)


def _worker_settings(params):
    worker_interface = params.get("workerInterface", {})
    algorithm = annotation_tools.get_required_select(
        worker_interface.get("Illumination algorithm"),
        "Illumination algorithm",
        allowed_values=ALGORITHM_OPTIONS,
    )
    refine = worker_interface.get("Refine stitch positions")
    if not isinstance(refine, bool):
        raise ValueError("'Refine stitch positions' must be selected or cleared")
    channel = worker_interface.get("Refinement channel")
    if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)):
        raise ValueError("'Refinement channel' must identify one channel")
    threshold = worker_interface.get("NCC threshold")
    if isinstance(threshold, bool) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        raise ValueError("'NCC threshold' must be numeric")
    threshold = float(threshold)
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("'NCC threshold' must be between 0.5 and 1.0")
    output_name = worker_interface.get("Output filename", "")
    if not isinstance(output_name, str):
        raise ValueError("'Output filename' must be text")
    output_name = output_name.strip()
    if output_name and Path(output_name).name != output_name:
        raise ValueError("'Output filename' must be a filename, not a path")
    if output_name and Path(output_name).suffix.lower() not in {".tif", ".tiff"}:
        output_name += ".tiff"
    return {
        "algorithm": algorithm,
        "adaptive_tile_gains": algorithm == ALGORITHM_OPTIONS[0],
        "refine": refine,
        "refinement_channel": int(channel),
        "ncc_threshold": threshold,
        "output_name": output_name,
    }


def compute(datasetId, apiUrl, token, params):
    try:
        settings = _worker_settings(params)
        girder = GirderClient(apiUrl=apiUrl)
        girder.setToken(token)
        sendProgress(
            0.01,
            WORKER_NAME,
            "Resolving multi-source geometry and the original ND2",
        )
        with tempfile.TemporaryDirectory(prefix="stitch-refinement-") as scratch_name:
            scratch = Path(scratch_name)
            downloaded = download_composite_sources(girder, datasetId, scratch)
            sendProgress(
                0.12,
                WORKER_NAME,
                "Loading raw reference tiles and camera-coordinate training planes",
            )
            with ND2File(downloaded.source_path) as raw_file:
                training = load_training_data(
                    raw_file,
                    settings["refinement_channel"],
                    model_size=MODEL_SIZE,
                )
                layout = parse_source_layout(
                    downloaded.document,
                    positions=training.positions,
                    time_points=training.time_points,
                    z_planes=training.z_planes,
                    channels=training.channels,
                    loop_indices=raw_file.loop_indices,
                )

                def pair_progress(done, total, edge):
                    sendProgress(
                        0.24 + 0.20 * done / max(total, 1),
                        WORKER_NAME,
                        f"Matching adjacent raw tiles {done}/{total}",
                    )

                refinement = refine_positions(
                    training.reference_tiles,
                    layout.positions,
                    training.stage_positions_um,
                    linear_transform=layout.linear_transform,
                    ncc_threshold=settings["ncc_threshold"],
                    progress=pair_progress,
                )
                if not refinement.accepted_measurements:
                    raise ValueError(
                        "No adjacent tile pairs met the NCC threshold. Choose a "
                        "higher-texture refinement channel or verify the original ND2."
                    )
                if refinement.max_residual is not None and refinement.max_residual > 2.0:
                    sendWarning(
                        "The refined stitch has a residual above the 2 px quality gate.",
                        info=f"Maximum confident-pair residual: {refinement.max_residual:.2f} px",
                    )
                output_positions = (
                    refinement.positions
                    if settings["refine"]
                    else np.rint(layout.positions).astype(np.int64)
                )
                original_bounds = transformed_bounds(
                    layout.positions, layout.linear_transform, training.raw_shape
                )
                output_bounds = transformed_bounds(
                    output_positions, layout.linear_transform, training.raw_shape
                )
                bounds_delta = [
                    float(after - before)
                    for before, after in zip(
                        original_bounds, output_bounds, strict=True
                    )
                ]
                if max(abs(delta) for delta in bounds_delta) > 16.0:
                    sendWarning(
                        "The refined mosaic exceeds the 16 px coordinate-stability "
                        "target.",
                        info=(
                            f"Boundary changes are {bounds_delta}. Existing annotation "
                            "coordinates are not moved and may need adjustment."
                        ),
                    )

                models = []
                for channel in range(training.channels):
                    sendProgress(
                        0.45 + 0.14 * channel / max(training.channels, 1),
                        WORKER_NAME,
                        f"Fitting overlap-DCT illumination model for "
                        f"{training.channel_names[channel]}",
                    )
                    models.append(
                        fit_overlap_dct(
                            training.model_stacks[channel],
                            refinement.accepted_measurements,
                            raw_shape=training.raw_shape,
                            adaptive_tile_gains=settings["adaptive_tile_gains"],
                        )
                    )

                corrected_tiles_path = scratch / "corrected_tiles.tiff"

                def plane_progress(done, total):
                    if done == total or done % max(total // 100, 1) == 0:
                        sendProgress(
                            0.60 + 0.20 * done / max(total, 1),
                            WORKER_NAME,
                            f"Correcting raw tile planes {done}/{total}",
                        )

                write_corrected_tile_tiff(
                    raw_file,
                    corrected_tiles_path,
                    models,
                    positions=training.positions,
                    time_points=training.time_points,
                    z_planes=training.z_planes,
                    channels=training.channels,
                    progress=plane_progress,
                )

            corrected_document = corrected_source_document(
                downloaded.document,
                layout,
                output_positions,
                corrected_tiles_path.name,
            )
            corrected_document_path = scratch / "corrected-multi-source.json"
            corrected_document_path.write_text(
                json.dumps(corrected_document, indent=2) + "\n", encoding="utf-8"
            )
            output_name = settings["output_name"] or (
                f"{downloaded.source_path.stem}_stitch_refined_"
                "illumination_corrected.tiff"
            )
            output_path = scratch / output_name
            sendProgress(
                0.82,
                WORKER_NAME,
                "Streaming corrected tiles into a pyramidal TIFF",
            )
            convert_multi_source(corrected_document_path, output_path)

            refinement_report = refinement.as_dict()
            refinement_report.update(
                {
                    "position_adjustment_enabled": settings["refine"],
                    "max_position_shift_px": float(
                        np.max(np.abs(output_positions - layout.positions))
                    ),
                    "original_bounds": list(original_bounds),
                    "output_bounds": list(output_bounds),
                    "bounds_delta_px": bounds_delta,
                }
            )
            metadata = {
                "tool": WORKER_NAME,
                "worker_version": WORKER_VERSION,
                "parameters": {
                    "refine_stitch_positions": settings["refine"],
                    "refinement_channel_zero_based": settings["refinement_channel"],
                    "refinement_channel_name": training.channel_names[
                        settings["refinement_channel"]
                    ],
                    "ncc_threshold": settings["ncc_threshold"],
                    "illumination_algorithm": settings["algorithm"],
                    "flatfield_model_size": MODEL_SIZE,
                    "flatfield_reference_z_zero_based": training.home_z,
                },
                "source": {
                    "multi_source_item_id": downloaded.document_item_id,
                    "original_nd2_item_id": downloaded.source_item_id,
                    "original_nd2_name": downloaded.source_path.name,
                },
                "refinement": refinement_report,
                "illumination": {
                    name: model.diagnostics
                    for name, model in zip(
                        training.channel_names, models, strict=True
                    )
                },
            }
            sendProgress(0.95, WORKER_NAME, "Uploading the new corrected image")
            item = upload_result(
                girder, datasetId, output_path, output_name, metadata
            )
            sendProgress(
                1.0,
                WORKER_NAME,
                f"Uploaded {output_name}; {len(refinement.accepted_measurements)}/"
                f"{len(refinement.measurements)} pairs matched, maximum residual "
                f"{refinement.max_residual:.2f} px",
            )
            print(json.dumps({"item": item, "metadata": metadata}, sort_keys=True))
            return item
    except RawCompositeRequiredError as exc:
        sendError("Raw composite input required.", info=str(exc))
        raise
    except Exception as exc:
        sendError(f"{WORKER_NAME} failed.", info=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=WORKER_NAME)
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
