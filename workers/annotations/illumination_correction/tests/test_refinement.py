from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from refinement import (  # noqa: E402
    build_adjacency,
    measure_pair,
    refine_positions,
)


def test_seeded_pair_search_recovers_known_shift() -> None:
    rng = np.random.default_rng(20260830)
    scene = ndimage.gaussian_filter(
        rng.normal(size=(140, 200)).astype(np.float32), sigma=1.5
    )
    first = scene[8:104, 10:106]
    second = scene[11:107, 75:171]

    measurement = measure_pair(
        first,
        second,
        first_index=3,
        second_index=4,
        predicted_shift=(62, 1),
    )

    assert (measurement.shift_x, measurement.shift_y) == (65, 3)
    assert measurement.ncc > 0.999
    assert measurement.accepted


def test_grid_adjacency_uses_stage_bins_not_acquisition_order() -> None:
    # Serpentine acquisition order: the middle row runs right-to-left.
    stages = np.asarray(
        [
            (0.0, 0.0),
            (100.0, 0.0),
            (200.0, 0.0),
            (200.0, 100.0),
            (100.0, 100.0),
            (0.0, 100.0),
        ]
    )

    edges = build_adjacency(stages)
    pairs = {(min(edge.first, edge.second), max(edge.first, edge.second)) for edge in edges}

    assert pairs == {
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (0, 5),
        (1, 4),
        (2, 3),
    }


def test_global_refinement_recovers_synthetic_tile_positions() -> None:
    rng = np.random.default_rng(151)
    tile_size = 112
    pitch = 64
    rows = 3
    columns = 3
    margin = 8
    scene = ndimage.gaussian_filter(
        rng.normal(size=(300, 300)).astype(np.float32), sigma=1.5
    )
    displacements = np.asarray(
        [
            (-2, 1),
            (0, 0),
            (2, -1),
            (-1, -2),
            (1, 2),
            (3, 0),
            (-3, 1),
            (0, -1),
            (2, 0),
        ],
        dtype=np.int64,
    )

    tiles = []
    stage_positions = []
    nominal_origins = []
    actual_origins = []
    for row in range(rows):
        column_order = range(columns) if row % 2 == 0 else range(columns - 1, -1, -1)
        for column in column_order:
            index = row * columns + (column if row % 2 == 0 else columns - 1 - column)
            nominal = np.asarray((column * pitch, row * pitch), dtype=np.int64)
            actual = nominal + displacements[index]
            x0, y0 = actual + margin
            tiles.append(scene[y0 : y0 + tile_size, x0 : x0 + tile_size])
            stage_positions.append((column * 100.0, row * 100.0))
            nominal_origins.append(nominal)
            actual_origins.append(actual)

    nominal_origins = np.asarray(nominal_origins)
    actual_origins = np.asarray(actual_origins)
    original_positions = 500 - nominal_origins
    true_positions = 500 - actual_origins
    true_positions = true_positions + np.mean(
        original_positions - true_positions, axis=0, keepdims=True
    )

    result = refine_positions(
        tiles,
        original_positions,
        np.asarray(stage_positions),
    )

    assert len(result.measurements) == 12
    assert len(result.accepted_measurements) == 12
    np.testing.assert_allclose(result.positions, true_positions, atol=1.0)
    assert result.max_residual <= 2.0
    assert np.max(np.abs(np.mean(result.positions - original_positions, axis=0))) <= 0.5


def test_low_texture_pair_is_dropped_and_does_not_move_tiles() -> None:
    tiles = [
        np.full((64, 64), 100.0, dtype=np.float32),
        np.full((64, 64), 100.0, dtype=np.float32),
    ]
    original_positions = np.asarray(((200, 200), (150, 200)), dtype=np.float64)
    stage_positions = np.asarray(((0.0, 0.0), (100.0, 0.0)), dtype=np.float64)

    result = refine_positions(tiles, original_positions, stage_positions)

    assert len(result.measurements) == 1
    assert not result.measurements[0].accepted
    assert result.measurements[0].ncc < 0.5
    assert result.accepted_measurements == ()
    np.testing.assert_array_equal(result.positions, original_positions.astype(np.int64))
    assert result.max_residual is None
