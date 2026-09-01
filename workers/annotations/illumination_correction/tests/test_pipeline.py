from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from illumination import IlluminationModel  # noqa: E402
from pipeline import (  # noqa: E402
    RawCompositeRequiredError,
    _exact_items,
    _item_files,
    corrected_source_document,
    convert_multi_source,
    download_composite_sources,
    load_training_data,
    parse_source_layout,
    transformed_bounds,
    write_corrected_tile_tiff,
)


class _PaginatedClient:
    def listItem(self, folder_id, **kwargs):
        assert "limit" not in kwargs
        assert kwargs == {"name": "multi-source2.json"}
        return iter(
            (
                {"_id": "wanted", "name": "multi-source2.json"},
                {"_id": "other", "name": "other.json"},
            )
        )

    def listFile(self, item_id, **kwargs):
        assert item_id == "wanted"
        assert "limit" not in kwargs
        return iter(({"_id": "file", "name": "multi-source2.json"},))


def test_girder_resource_helpers_use_exhaustive_default_pagination() -> None:
    client = _PaginatedClient()

    assert _exact_items(client, "folder", "multi-source2.json") == [
        {"_id": "wanted", "name": "multi-source2.json"}
    ]
    assert _item_files(client, "wanted") == [
        {"_id": "file", "name": "multi-source2.json"}
    ]


class _StitchedOnlyClient:
    def listItem(self, folder_id, **kwargs):
        assert folder_id == "stitched-folder"
        assert kwargs == {"name": "multi-source2.json"}
        return iter(())


def test_source_discovery_explains_that_stitched_only_images_are_unsupported(
    tmp_path,
) -> None:
    try:
        download_composite_sources(
            _StitchedOnlyClient(), "stitched-folder", tmp_path
        )
    except RawCompositeRequiredError as exc:
        message = str(exc)
        assert "already-stitched TIFF-only" in message
        assert "raw tile overlaps" in message
        assert "multi-source2.json" in message
    else:
        raise AssertionError("a stitched-only dataset should be rejected")


class _TiffBackedCompositeClient:
    def listItem(self, folder_id, **kwargs):
        assert folder_id == "derived-folder"
        assert kwargs == {"name": "multi-source2.json"}
        return iter(({"_id": "document", "name": "multi-source2.json"},))

    def listFile(self, item_id, **kwargs):
        assert item_id == "document"
        assert kwargs == {}
        return iter(({"_id": "document-file", "name": "multi-source2.json"},))

    def downloadFile(self, file_id, destination):
        assert file_id == "document-file"
        Path(destination).write_text(
            json.dumps({"sources": [{"path": "already-stitched.tiff"}]}),
            encoding="utf-8",
        )


def test_source_discovery_rejects_a_multisource_document_backed_by_tiff(
    tmp_path,
) -> None:
    try:
        download_composite_sources(
            _TiffBackedCompositeClient(), "derived-folder", tmp_path
        )
    except RawCompositeRequiredError as exc:
        message = str(exc)
        assert "already-stitched.tiff" in message
        assert "not an original ND2" in message
        assert "raw tile overlaps are unavailable" in message
    else:
        raise AssertionError("a TIFF-backed composite should be rejected")


P_MAJOR_LOOP_INDICES = tuple(
    {"P": position, "T": 0, "Z": z_index}
    for position in range(2)
    for z_index in range(2)
)


def _document(paths=("source.nd2",)):
    sources = []
    positions = ((100, 200), (60, 200))
    for sequence_index, indices in enumerate(P_MAJOR_LOOP_INDICES):
        position_index = indices["P"]
        for channel in range(2):
            sources.append(
                {
                    "path": paths[position_index % len(paths)],
                    "xySet": 0,
                    "zSet": indices["Z"],
                    "tSet": indices["T"],
                    "cSet": channel,
                    "frames": [sequence_index * 2 + channel],
                    "position": {
                        "x": positions[position_index][0],
                        "y": positions[position_index][1],
                        "s11": -1,
                        "s12": 0,
                        "s21": 0,
                        "s22": -1,
                    },
                }
            )
    return {"channels": ["A", "B"], "sources": sources}


def test_source_document_updates_translations_only() -> None:
    document = _document()
    layout = parse_source_layout(
        document,
        positions=2,
        z_planes=2,
        channels=2,
        loop_indices=P_MAJOR_LOOP_INDICES,
    )
    refined = np.asarray(((103, 198), (58, 202)))

    output = corrected_source_document(
        document, layout, refined, "corrected_tiles.tiff"
    )

    assert output is not document
    assert {source["path"] for source in output["sources"]} == {
        "corrected_tiles.tiff"
    }
    assert {source["sourceName"] for source in output["sources"]} == {"tiff"}
    assert output["sources"][0]["position"] == {
        "x": 103,
        "y": 198,
        "s11": -1,
        "s12": 0,
        "s21": 0,
        "s22": -1,
    }
    assert output["sources"][4]["position"]["x"] == 58
    assert document["sources"][0]["path"] == "source.nd2"


def test_source_document_preserves_fractional_translations() -> None:
    document = _document()
    layout = parse_source_layout(
        document,
        positions=2,
        z_planes=2,
        channels=2,
        loop_indices=P_MAJOR_LOOP_INDICES,
    )
    deployed = np.asarray(((100.25, 200.75), (60.5, 199.125)))

    output = corrected_source_document(
        document, layout, deployed, "corrected_tiles.tiff"
    )

    assert output["sources"][0]["position"]["x"] == 100.25
    assert output["sources"][0]["position"]["y"] == 200.75
    assert output["sources"][4]["position"]["x"] == 60.5
    assert output["sources"][4]["position"]["y"] == 199.125
    json.dumps(output)


def test_source_layout_uses_nd2_position_indices_for_time_major_frames() -> None:
    positions = ((100, 200), (60, 200))
    loop_indices = (
        {"T": 0, "P": 0, "Z": 0},
        {"T": 0, "P": 1, "Z": 0},
        {"T": 1, "P": 0, "Z": 0},
        {"T": 1, "P": 1, "Z": 0},
    )
    sources = []
    for sequence_index, indices in enumerate(loop_indices):
        for channel in range(2):
            position = positions[indices["P"]]
            sources.append(
                {
                    "path": "source.nd2",
                    "xySet": 0,
                    "tSet": indices["T"],
                    "zSet": indices["Z"],
                    "cSet": channel,
                    "frames": [sequence_index * 2 + channel],
                    "position": {
                        "x": position[0],
                        "y": position[1],
                        "s11": -1,
                        "s12": 0,
                        "s21": 0,
                        "s22": -1,
                    },
                }
            )

    layout = parse_source_layout(
        {"channels": ["A", "B"], "sources": sources},
        positions=2,
        time_points=2,
        z_planes=1,
        channels=2,
        loop_indices=loop_indices,
    )

    np.testing.assert_array_equal(layout.positions, positions)
    assert layout.source_position_indices == (0, 0, 1, 1, 0, 0, 1, 1)


class _TimeLapseRawFile:
    sizes = {"T": 3, "P": 2, "Z": 2, "C": 2, "Y": 4, "X": 4}
    loop_indices = tuple(
        {"T": time, "P": position, "Z": z_index}
        for time in range(3)
        for position in range(2)
        for z_index in range(2)
    )
    experiment = ()
    metadata = SimpleNamespace(
        channels=(
            SimpleNamespace(channel=SimpleNamespace(name="DAPI")),
            SimpleNamespace(channel=SimpleNamespace(name="YFP")),
        )
    )

    def __init__(self):
        self.read_calls = []

    def asarray(self, position):
        raise AssertionError("training must not materialize every time point")

    def read_frame(self, sequence_index):
        self.read_calls.append(sequence_index)
        indices = self.loop_indices[sequence_index]
        base = 100 * indices["P"] + 10 * indices["T"] + indices["Z"]
        return np.stack(
            (
                np.full((4, 4), base, dtype=np.uint16),
                np.full((4, 4), 1000 + base, dtype=np.uint16),
            )
        )

    def frame_metadata(self, sequence_index):
        position = self.loop_indices[sequence_index]["P"]
        stage = SimpleNamespace(x=100.0 * position, y=10.0 * position)
        return SimpleNamespace(
            channels=(
                SimpleNamespace(
                    position=SimpleNamespace(stagePositionUm=stage)
                ),
            )
        )


def test_training_reads_only_time_zero_sequences() -> None:
    raw_file = _TimeLapseRawFile()

    training = load_training_data(raw_file, refinement_channel=0, model_size=4)

    assert raw_file.read_calls == [0, 1, 2, 3]
    assert all(raw_file.loop_indices[index]["T"] == 0 for index in raw_file.read_calls)
    assert [int(tile[0, 0]) for tile in training.reference_tiles] == [1, 101]
    assert int(training.model_stacks[0, 0, 0, 0]) == 1
    assert int(training.model_stacks[1, 1, 0, 0]) == 1101
    np.testing.assert_array_equal(
        training.stage_positions_um, ((0.0, 0.0), (100.0, 10.0))
    )


def test_source_document_rejects_multiple_original_files() -> None:
    try:
        parse_source_layout(
            _document(paths=("first.nd2", "second.nd2")),
            positions=2,
            z_planes=2,
            channels=2,
            loop_indices=P_MAJOR_LOOP_INDICES,
        )
    except ValueError as exc:
        assert "one ND2" in str(exc)
    else:
        raise AssertionError("multiple ND2 paths should be rejected")


class _RawFile:
    def read_frame(self, sequence_index):
        return np.stack(
            (
                np.full((8, 8), 10 + sequence_index, dtype=np.uint16),
                np.full((8, 8), 20 + sequence_index, dtype=np.uint16),
            )
        )


def test_corrected_tile_tiff_preserves_raw_frame_order(tmp_path) -> None:
    output = tmp_path / "corrected.tiff"
    model = IlluminationModel(
        flatfield=np.ones((8, 8), dtype=np.float32),
        gains=np.asarray((1.0, 2.0), dtype=np.float32),
        diagnostics={},
    )

    write_corrected_tile_tiff(
        _RawFile(),
        output,
        (model, None),
        positions=2,
        z_planes=2,
        channels=2,
        tile_size=16,
    )

    with tifffile.TiffFile(output) as tiff:
        assert len(tiff.pages) == 8
        values = [int(page.asarray()[0, 0]) for page in tiff.pages]
        frame_axes = [json.loads(page.description)["frame"] for page in tiff.pages]
    assert values == [10, 20, 11, 21, 6, 22, 6, 23]
    assert frame_axes == [
        {"IndexC": channel, "IndexT": 0, "IndexZ": z_index}
        for _position in range(2)
        for z_index in range(2)
        for channel in range(2)
    ]


def test_bounds_use_the_preserved_linear_transform() -> None:
    positions = np.asarray(((100, 100), (50, 100)))
    bounds = transformed_bounds(positions, -np.eye(2), (20, 30))
    assert bounds == (20.0, 80.0, 100.0, 100.0)


def test_converter_streams_corrected_tile_multisource_to_pyramid(tmp_path) -> None:
    tiles_path = tmp_path / "corrected_tiles.tiff"
    values = (11, 101, 22, 202)
    with tifffile.TiffWriter(tiles_path, bigtiff=True) as writer:
        for frame, value in enumerate(values):
            writer.write(
                np.full((32, 32), value, dtype=np.uint16),
                photometric="minisblack",
                tile=(16, 16),
                description=json.dumps(
                    {
                        "frame": {
                            "IndexC": frame % 2,
                            "IndexT": 0,
                            "IndexZ": 0,
                        }
                    }
                ),
                metadata=None,
            )
    document = {
        "channels": ["DAPI", "YFP"],
        "singleBand": False,
        "uniformSources": True,
        "sources": [
            {
                "path": tiles_path.name,
                "sourceName": "tiff",
                "xySet": 0,
                "zSet": 0,
                "tSet": 0,
                "cSet": frame % 2,
                "frames": [frame],
                "position": {
                    "x": 32 + (frame // 2) * 32,
                    "y": 32,
                    "s11": -1,
                    "s12": 0,
                    "s21": 0,
                    "s22": -1,
                },
            }
            for frame in range(4)
        ],
    }
    document_path = tmp_path / "corrected-multi-source.json"
    document_path.write_text(json.dumps(document), encoding="utf-8")
    output_path = tmp_path / "converted.tiff"

    convert_multi_source(document_path, output_path, tile_size=32)

    with tifffile.TiffFile(output_path) as tiff:
        assert tiff.pages[0].is_tiled
        image = tiff.pages[0].asarray()
    assert image.shape[:2] == (32, 64)
    assert {11, 22}.issubset(set(np.unique(image)))
