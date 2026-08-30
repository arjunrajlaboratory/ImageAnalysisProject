from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from illumination import IlluminationModel  # noqa: E402
from pipeline import (  # noqa: E402
    _exact_items,
    _item_files,
    corrected_source_document,
    convert_multi_source,
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


def _document(paths=("source.nd2",)):
    sources = []
    positions = ((100, 200), (60, 200))
    for frame in range(8):
        position = frame // 4
        sources.append(
            {
                "path": paths[position % len(paths)],
                "zSet": (frame % 4) // 2,
                "cSet": frame % 2,
                "frames": [frame],
                "position": {
                    "x": positions[position][0],
                    "y": positions[position][1],
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
    layout = parse_source_layout(document, positions=2, z_planes=2, channels=2)
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


def test_source_document_rejects_multiple_original_files() -> None:
    try:
        parse_source_layout(
            _document(paths=("first.nd2", "second.nd2")),
            positions=2,
            z_planes=2,
            channels=2,
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
