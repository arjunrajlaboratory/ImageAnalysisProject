from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from illumination import aligned_slices, fit_overlap_dct  # noqa: E402
from refinement import PairMeasurement, build_adjacency  # noqa: E402


def _median_overlap_log_bias(stack, measurements):
    values = []
    for measurement in measurements:
        slices = aligned_slices(
            stack.shape[-2:], measurement.shift_x, measurement.shift_y
        )
        first = stack[measurement.first][slices[0]]
        second = stack[measurement.second][slices[1]]
        values.append(
            abs(
                float(
                    np.median(
                        np.log(np.maximum(first, 1.0))
                        - np.log(np.maximum(second, 1.0))
                    )
                )
            )
        )
    return float(np.median(values))


def test_overlap_dct_reduces_synthetic_vignetting_and_tile_gain_bias() -> None:
    rng = np.random.default_rng(31)
    tile_size = 96
    pitch = 64
    scene = 2500.0 + 500.0 * ndimage.gaussian_filter(
        rng.normal(size=(240, 240)).astype(np.float32), sigma=1.2
    )
    coordinate = np.linspace(-1.0, 1.0, tile_size, dtype=np.float32)
    y_coordinate = coordinate[:, None]
    x_coordinate = coordinate[None, :]
    true_flat = np.exp(
        -0.22 * x_coordinate**2
        - 0.15 * y_coordinate**2
        + 0.12 * x_coordinate
        + 0.05 * x_coordinate * y_coordinate
    )
    true_flat /= np.mean(true_flat)
    true_gains = np.exp(np.linspace(-0.006, 0.006, 9)).astype(np.float32)
    true_gains /= np.mean(true_gains)

    tiles = []
    stages = []
    origins = []
    for row in range(3):
        order = range(3) if row % 2 == 0 else range(2, -1, -1)
        for column in order:
            x0 = column * pitch
            y0 = row * pitch
            tiles.append(scene[y0 : y0 + tile_size, x0 : x0 + tile_size])
            stages.append((column * 100.0, row * 100.0))
            origins.append((x0, y0))
    origins = np.asarray(origins)
    observed = np.stack(
        [
            tile * true_flat * true_gains[index]
            for index, tile in enumerate(tiles)
        ]
    ).astype(np.float32)
    measurements = []
    for edge in build_adjacency(np.asarray(stages)):
        shift = origins[edge.second] - origins[edge.first]
        measurements.append(
            PairMeasurement(
                first=edge.first,
                second=edge.second,
                axis=edge.axis,
                predicted_shift_x=int(shift[0]),
                predicted_shift_y=int(shift[1]),
                shift_x=int(shift[0]),
                shift_y=int(shift[1]),
                ncc=0.99,
                accepted=True,
            )
        )

    model = fit_overlap_dct(observed, measurements)
    corrected = model.apply_stack(observed)

    raw_bias = _median_overlap_log_bias(observed, measurements)
    corrected_bias = _median_overlap_log_bias(corrected, measurements)
    relative_flat_error = np.median(np.abs(model.flatfield / true_flat - 1.0))

    assert corrected_bias < raw_bias * 0.35
    assert relative_flat_error < 0.08
    assert model.diagnostics["method"] == "overlap_dct_gain"
    assert model.diagnostics["used_pairs"] == 12
    assert np.all(np.isfinite(corrected))


def test_overlap_dct_can_disable_adaptive_tile_gains() -> None:
    stack = np.stack(
        [np.full((48, 48), 1000.0 + index, dtype=np.float32) for index in range(2)]
    )
    measurement = PairMeasurement(
        first=0,
        second=1,
        axis="horizontal",
        predicted_shift_x=16,
        predicted_shift_y=0,
        shift_x=16,
        shift_y=0,
        ncc=0.9,
        accepted=True,
    )

    model = fit_overlap_dct(
        stack, [measurement], adaptive_tile_gains=False
    )

    np.testing.assert_array_equal(model.gains, np.ones(2, dtype=np.float32))
    assert model.diagnostics["method"] == "overlap_dct"
