"""Girder, ND2, corrected-tile, and multi-source plumbing for the worker."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
from girder_client import GirderClient
from nd2 import ND2File
from skimage.transform import resize

from illumination import IlluminationModel


MULTI_SOURCE_NAME = "multi-source2.json"
MODEL_SIZE = 128
UINT16_MAX = np.iinfo(np.uint16).max


@dataclass(frozen=True)
class SourceLayout:
    source_path: str
    positions: np.ndarray
    linear_transform: np.ndarray
    source_position_indices: tuple[int, ...]
    frames_per_position: int


@dataclass(frozen=True)
class DownloadedSources:
    document: dict
    document_path: Path
    document_item_id: str
    source_path: Path
    source_item_id: str


@dataclass(frozen=True)
class RawTrainingData:
    reference_tiles: tuple[np.ndarray, ...]
    model_stacks: np.ndarray
    stage_positions_um: np.ndarray
    channel_names: tuple[str, ...]
    raw_shape: tuple[int, int]
    positions: int
    time_points: int
    z_planes: int
    channels: int
    home_z: int


def _item_files(client: GirderClient, item_id: str) -> list[dict]:
    return list(client.listFile(item_id))


def _exact_items(client: GirderClient, folder_id: str, name: str) -> list[dict]:
    return [
        item
        for item in client.listItem(folder_id, name=name)
        if item.get("name") == name
    ]


def download_composite_sources(
    client: GirderClient, dataset_id: str, scratch: Path
) -> DownloadedSources:
    """Download the deployed multi-source JSON and its one original ND2."""
    document_items = _exact_items(client, dataset_id, MULTI_SOURCE_NAME)
    if len(document_items) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {MULTI_SOURCE_NAME!r} item in dataset folder "
            f"{dataset_id}, found {len(document_items)}. This worker only supports "
            "single-file composited datasets."
        )
    document_item = document_items[0]
    document_files = [
        file
        for file in _item_files(client, document_item["_id"])
        if file.get("name") == MULTI_SOURCE_NAME
    ]
    if len(document_files) != 1:
        raise FileNotFoundError(
            f"The {MULTI_SOURCE_NAME!r} item does not contain its source JSON file."
        )
    document_path = scratch / MULTI_SOURCE_NAME
    client.downloadFile(document_files[0]["_id"], str(document_path))
    document = json.loads(document_path.read_text(encoding="utf-8"))
    paths = {
        str(source.get("path", ""))
        for source in document.get("sources", [])
        if source.get("path")
    }
    if len(paths) != 1:
        raise ValueError(
            "v1 requires every multi-source entry to point to one original ND2; "
            f"found {len(paths)} distinct paths"
        )
    source_name = Path(next(iter(paths))).name
    if Path(source_name).suffix.lower() != ".nd2":
        raise ValueError(
            f"v1 supports an original ND2 source, but the stitch references {source_name!r}"
        )
    source_items = _exact_items(client, dataset_id, source_name)
    source_files = []
    for item in source_items:
        for file in _item_files(client, item["_id"]):
            if file.get("name") == source_name:
                source_files.append((item, file))
    if len(source_files) != 1:
        raise FileNotFoundError(
            f"The original ND2 {source_name!r} referenced by {MULTI_SOURCE_NAME} "
            "is missing. The worker cannot reconstruct raw tiles after that original "
            "item has been deleted."
        )
    source_item, source_file = source_files[0]
    source_path = scratch / source_name
    client.downloadFile(source_file["_id"], str(source_path))
    return DownloadedSources(
        document=document,
        document_path=document_path,
        document_item_id=str(document_item["_id"]),
        source_path=source_path,
        source_item_id=str(source_item["_id"]),
    )


def parse_source_layout(
    document: dict,
    *,
    positions: int,
    time_points: int = 1,
    z_planes: int,
    channels: int,
) -> SourceLayout:
    """Validate the single-ND2 source/frame contract and recover seeded geometry."""
    sources = document.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("multi-source document has no sources")
    frames_per_position = int(time_points) * int(z_planes) * int(channels)
    expected_sources = int(positions) * frames_per_position
    if len(sources) != expected_sources:
        raise ValueError(
            f"expected {expected_sources} P×T×Z×C source records, found {len(sources)}"
        )
    paths = {str(source.get("path", "")) for source in sources}
    if len(paths) != 1 or not next(iter(paths)):
        raise ValueError("v1 requires all source records to reference one ND2 path")

    by_position: list[dict | None] = [None] * positions
    source_position_indices = []
    seen_frames = set()
    shared_transform = None
    for source_index, source in enumerate(sources):
        frames = source.get("frames")
        if not isinstance(frames, list) or len(frames) != 1:
            raise ValueError(
                f"source {source_index} must identify exactly one raw frame"
            )
        frame = int(frames[0])
        if frame < 0 or frame >= expected_sources or frame in seen_frames:
            raise ValueError(f"source {source_index} has invalid or duplicate frame {frame}")
        seen_frames.add(frame)
        position_index = frame // frames_per_position
        source_position_indices.append(position_index)
        position = source.get("position")
        required = ("x", "y", "s11", "s12", "s21", "s22")
        if not isinstance(position, dict) or any(key not in position for key in required):
            raise ValueError(f"source {source_index} has incomplete stitch geometry")
        transform = np.asarray(
            (
                (position["s11"], position["s12"]),
                (position["s21"], position["s22"]),
            ),
            dtype=np.float64,
        )
        if shared_transform is None:
            shared_transform = transform
        elif not np.allclose(transform, shared_transform, rtol=0.0, atol=1e-9):
            raise ValueError(
                "v1 requires one shared camera transform across every raw tile"
            )
        prior = by_position[position_index]
        if prior is None:
            by_position[position_index] = position
        elif any(float(prior[key]) != float(position[key]) for key in required):
            raise ValueError(
                f"raw position {position_index} has inconsistent source transforms"
            )
    if seen_frames != set(range(expected_sources)) or any(item is None for item in by_position):
        raise ValueError("multi-source frames do not cover every raw P×T×Z×C frame")
    translations = np.asarray(
        [(item["x"], item["y"]) for item in by_position], dtype=np.float64
    )
    if not np.all(np.isfinite(translations)):
        raise ValueError("multi-source translations must be finite")
    if abs(float(np.linalg.det(shared_transform))) <= np.finfo(np.float64).eps:
        raise ValueError("the shared camera transform is singular")
    return SourceLayout(
        source_path=next(iter(paths)),
        positions=translations,
        linear_transform=shared_transform,
        source_position_indices=tuple(source_position_indices),
        frames_per_position=frames_per_position,
    )


def corrected_source_document(
    document: dict,
    layout: SourceLayout,
    refined_positions: np.ndarray,
    corrected_tiles_name: str,
) -> dict:
    """Point each source frame at the corrected TIFF and change translations only."""
    positions = np.asarray(refined_positions)
    if positions.shape != layout.positions.shape:
        raise ValueError(
            f"refined positions must have shape {layout.positions.shape}, got {positions.shape}"
        )
    output = copy.deepcopy(document)
    for source, position_index in zip(
        output["sources"], layout.source_position_indices, strict=True
    ):
        source["path"] = str(corrected_tiles_name)
        source["sourceName"] = "tiff"
        # Deliberately preserve s11..s22 verbatim from the deployed geometry.
        source["position"]["x"] = int(positions[position_index, 0])
        source["position"]["y"] = int(positions[position_index, 1])
    return output


def _home_z_index(raw_file: ND2File, z_planes: int) -> int:
    indices = [
        int(loop.parameters.homeIndex)
        for loop in raw_file.experiment
        if type(loop).__name__ == "ZStackLoop"
    ]
    if len(indices) == 1 and 0 <= indices[0] < z_planes:
        return indices[0]
    return z_planes // 2


def load_training_data(
    raw_file: ND2File,
    refinement_channel: int,
    *,
    model_size: int = MODEL_SIZE,
) -> RawTrainingData:
    sizes = {key: int(value) for key, value in raw_file.sizes.items()}
    missing = [axis for axis in ("P", "C", "Y", "X") if axis not in sizes]
    if missing:
        raise ValueError(f"raw ND2 lacks required axes {missing}: {sizes}")
    positions = sizes["P"]
    time_points = int(sizes.get("T", 1))
    z_planes = int(sizes.get("Z", 1))
    channels = sizes["C"]
    height = sizes["Y"]
    width = sizes["X"]
    if not 0 <= int(refinement_channel) < channels:
        raise ValueError(
            f"refinement channel {refinement_channel} is outside 0..{channels - 1}"
        )
    channel_names = tuple(
        str(channel.channel.name) for channel in raw_file.metadata.channels
    )
    if len(channel_names) != channels:
        channel_names = tuple(f"Channel {index + 1}" for index in range(channels))
    home_z = _home_z_index(raw_file, z_planes)
    reference_tiles = []
    model_stacks = np.empty(
        (channels, positions, model_size, model_size), dtype=np.float32
    )
    stage_positions = []
    for position in range(positions):
        position_stack = np.asarray(raw_file.asarray(position))
        # nd2 retains a singleton P axis even after selecting a position.
        if position_stack.shape[0] != 1:
            raise ValueError(
                f"unexpected selected-position ND2 shape {position_stack.shape}"
            )
        position_stack = position_stack[0]
        if time_points > 1:
            if position_stack.ndim == 4 and z_planes == 1:
                position_stack = position_stack[:, None, ...]
            expected_position_shape = (
                time_points,
                z_planes,
                channels,
                height,
                width,
            )
            if position_stack.shape != expected_position_shape:
                raise ValueError(
                    f"expected selected-position ND2 shape {expected_position_shape}, "
                    f"found {position_stack.shape}"
                )
            position_stack = position_stack[0]
        elif position_stack.ndim == 3 and z_planes == 1:
            position_stack = position_stack[None, ...]
        expected_shape = (z_planes, channels, height, width)
        if position_stack.shape != expected_shape:
            raise ValueError(
                f"expected selected-position ND2 shape {expected_shape}, found "
                f"{position_stack.shape}"
            )
        reference_tiles.append(
            np.max(position_stack[:, refinement_channel], axis=0).astype(np.float32)
        )
        home_frame = position_stack[home_z]
        for channel in range(channels):
            model_stacks[channel, position] = resize(
                home_frame[channel],
                (model_size, model_size),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)
        # frame_metadata expects the raw sequence index, not the P index. In the
        # validated P×Z file this is position * Z; loop_indices also handles T.
        sequence_index = next(
            index
            for index, indices in enumerate(raw_file.loop_indices)
            if int(indices.get("P", 0)) == position
            and int(indices.get("T", 0)) == 0
            and int(indices.get("Z", 0)) == 0
        )
        metadata = raw_file.frame_metadata(sequence_index)
        stage = metadata.channels[0].position.stagePositionUm
        stage_positions.append((float(stage.x), float(stage.y)))
    return RawTrainingData(
        reference_tiles=tuple(reference_tiles),
        model_stacks=model_stacks,
        stage_positions_um=np.asarray(stage_positions, dtype=np.float64),
        channel_names=channel_names,
        raw_shape=(height, width),
        positions=positions,
        time_points=time_points,
        z_planes=z_planes,
        channels=channels,
        home_z=home_z,
    )


def write_corrected_tile_tiff(
    raw_file: ND2File,
    output_path: Path,
    models: Sequence[IlluminationModel | None],
    *,
    positions: int,
    time_points: int = 1,
    z_planes: int,
    channels: int,
    tile_size: int = 256,
    progress=None,
) -> None:
    if len(models) != channels:
        raise ValueError(f"expected {channels} channel models, found {len(models)}")
    if tile_size < 16 or tile_size % 16:
        raise ValueError("corrected-tile TIFF tile size must be a multiple of 16")
    total_planes = positions * time_points * z_planes * channels
    plane_index = 0
    loop_indices = getattr(raw_file, "loop_indices", None)
    if loop_indices is None:
        loop_indices = tuple(
            {"P": position, "T": time, "Z": z_index}
            for position in range(positions)
            for time in range(time_points)
            for z_index in range(z_planes)
        )
    expected_sequences = positions * time_points * z_planes
    if len(loop_indices) != expected_sequences:
        raise ValueError(
            f"expected {expected_sequences} raw P×T×Z frames, found {len(loop_indices)}"
        )
    with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
        for sequence_index, indices in enumerate(loop_indices):
            position = int(indices.get("P", 0))
            frame = np.asarray(raw_file.read_frame(sequence_index))
            if frame.ndim == 2 and channels == 1:
                frame = frame[None, ...]
            if frame.ndim != 3 or frame.shape[0] != channels:
                raise ValueError(
                    f"raw frame has shape {frame.shape}; expected (C, Y, X)"
                )
            for channel in range(channels):
                values = frame[channel]
                if models[channel] is not None:
                    values = models[channel].apply(values, position)
                output = np.rint(np.clip(values, 0.0, UINT16_MAX)).astype(np.uint16)
                writer.write(
                    output,
                    photometric="minisblack",
                    tile=(tile_size, tile_size),
                    description=json.dumps(
                        {
                            "frame": {
                                "IndexC": channel,
                                "IndexT": int(indices.get("T", 0)),
                                "IndexZ": int(indices.get("Z", 0)),
                            }
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    metadata=None,
                )
                plane_index += 1
                if progress is not None:
                    progress(plane_index, total_planes)


def convert_multi_source(
    source_document_path: Path, output_path: Path, *, tile_size: int = 512
) -> Path:
    """Stream a multi-source document through large_image_converter."""
    from large_image_converter import convert

    converted = convert(
        str(source_document_path),
        str(output_path),
        overwrite=True,
        compression="lzw",
        predictor="horizontal",
        tileSize=int(tile_size),
        _concurrency=1,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"large_image_converter did not create {output_path}")
    return Path(converted)


def upload_result(
    client: GirderClient,
    dataset_id: str,
    output_path: Path,
    output_name: str,
    metadata: dict,
) -> dict:
    item = client.uploadFileToFolder(
        dataset_id, str(output_path), filename=output_name
    )
    client.addMetadataToItem(item["itemId"], metadata)
    return item


def transformed_bounds(
    positions: np.ndarray,
    linear_transform: np.ndarray,
    raw_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Return min-x, min-y, max-x, max-y of all transformed raw tiles."""
    height, width = raw_shape
    corners = np.asarray(((0, 0), (width, 0), (0, height), (width, height)))
    transformed = corners @ np.asarray(linear_transform).T
    all_corners = transformed[None, :, :] + np.asarray(positions)[:, None, :]
    return (
        float(np.min(all_corners[:, :, 0])),
        float(np.min(all_corners[:, :, 1])),
        float(np.max(all_corners[:, :, 0])),
        float(np.max(all_corners[:, :, 1])),
    )
