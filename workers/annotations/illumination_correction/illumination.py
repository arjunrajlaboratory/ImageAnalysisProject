"""Raw-tile overlap-DCT flat-field correction supplied by the v7 study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage

from refinement import PairMeasurement


MIN_POSITIVE_SIGNAL = 1.0
PROFILE_SMOOTH_SIGMA = 3.0
OVERLAP_DCT_ORDER = 5
OVERLAP_DCT_RIDGE = 4.0
OVERLAP_DCT_IRLS_ITERATIONS = 6
HUBER_TUNING = 1.345
TILE_GAIN_RIDGE = 8.0
TILE_GAIN_MAX_FOLD = 1.10
OVERLAP_CHUNK_SIZE = 16
VALIDATION_LOW_PERCENTILE = 10.0
VALIDATION_HIGH_PERCENTILE = 90.0
MAX_DCT_SAMPLES = 250_000


@dataclass(frozen=True)
class IlluminationModel:
    flatfield: np.ndarray
    gains: np.ndarray
    diagnostics: dict

    def resized_flatfield(self, shape: tuple[int, int]) -> np.ndarray:
        field = np.asarray(self.flatfield, dtype=np.float32)
        if field.shape == tuple(shape):
            return field
        y_coordinates = np.linspace(0.0, field.shape[0] - 1.0, shape[0])
        x_coordinates = np.linspace(0.0, field.shape[1] - 1.0, shape[1])
        coordinates = np.stack(
            np.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        )
        resized = ndimage.map_coordinates(
            field, coordinates, order=3, mode="reflect"
        ).astype(np.float32)
        return normalize_flat(resized)

    def apply(self, image: np.ndarray, position: int) -> np.ndarray:
        values = np.asarray(image, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"expected one two-dimensional plane, got {values.shape}")
        if not 0 <= int(position) < self.gains.size:
            raise IndexError(position)
        flat = self.resized_flatfield(values.shape)
        corrected = values / (flat * float(self.gains[int(position)]))
        return corrected.astype(np.float32)

    def apply_stack(self, stack: np.ndarray) -> np.ndarray:
        values = np.asarray(stack, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != self.gains.size:
            raise ValueError(
                f"expected stack shape ({self.gains.size}, Y, X), got {values.shape}"
            )
        flat = self.resized_flatfield(values.shape[-2:])
        return (
            values / flat[None, :, :] / self.gains[:, None, None]
        ).astype(np.float32)


def normalize_flat(flatfield: np.ndarray) -> np.ndarray:
    values = np.asarray(flatfield, dtype=np.float32)
    mean = float(np.mean(values))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("flatfield has an invalid mean")
    values = values / mean
    if not np.all(np.isfinite(values)) or float(np.min(values)) <= 0.0:
        raise ValueError("flatfield must be finite and strictly positive")
    return values.astype(np.float32)


def fit_log_median(stack: np.ndarray) -> np.ndarray:
    """Fit the study's camera-coordinate log-median base field."""
    values = np.asarray(stack, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("flat-field fitting requires at least two tile images")
    log_values = np.log(np.maximum(values, MIN_POSITIVE_SIGNAL))
    log_values -= np.median(log_values, axis=(1, 2), keepdims=True)
    log_field = np.median(log_values, axis=0)
    log_field = ndimage.gaussian_filter(
        log_field, PROFILE_SMOOTH_SIGMA, mode="reflect"
    )
    return normalize_flat(np.exp(log_field))


def dct_basis(
    coordinates: np.ndarray, terms: Sequence[tuple[int, int]]
) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    y_coordinate = values[:, 0]
    x_coordinate = values[:, 1]
    return np.stack(
        [
            np.cos(np.pi * order_y * y_coordinate)
            * np.cos(np.pi * order_x * x_coordinate)
            for order_y, order_x in terms
        ],
        axis=1,
    )


def robust_ridge(
    design: np.ndarray,
    response: np.ndarray,
    penalty: np.ndarray,
    iterations: int = OVERLAP_DCT_IRLS_ITERATIONS,
) -> tuple[np.ndarray, dict[str, float]]:
    weights = np.ones(response.shape[0], dtype=np.float64)
    coefficient = np.zeros(design.shape[1], dtype=np.float64)
    residual = response.copy()
    scale = float("nan")
    for _ in range(int(iterations)):
        root_weight = np.sqrt(weights)
        weighted_design = design * root_weight[:, None]
        weighted_response = response * root_weight
        system = weighted_design.T @ weighted_design + np.diag(penalty)
        target = weighted_design.T @ weighted_response
        coefficient = np.linalg.solve(system, target)
        residual = response - design @ coefficient
        scale = max(
            1.4826 * float(np.median(np.abs(residual))),
            np.finfo(np.float64).eps,
        )
        weights = np.minimum(
            1.0,
            HUBER_TUNING
            * scale
            / np.maximum(np.abs(residual), np.finfo(np.float64).eps),
        )
    return coefficient, {
        "residual_median_abs": float(np.median(np.abs(residual))),
        "residual_scale": scale,
        "minimum_weight": float(np.min(weights)),
    }


def aligned_slices(
    shape: tuple[int, int], shift_x: int, shift_y: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    height, width = shape
    first_x0 = max(0, int(shift_x))
    first_x1 = min(width, int(shift_x) + width)
    first_y0 = max(0, int(shift_y))
    first_y1 = min(height, int(shift_y) + height)
    if first_x1 <= first_x0 or first_y1 <= first_y0:
        return None
    return (
        (slice(first_y0, first_y1), slice(first_x0, first_x1)),
        (
            slice(first_y0 - shift_y, first_y1 - shift_y),
            slice(first_x0 - shift_x, first_x1 - shift_x),
        ),
    )


def _scaled_shift(
    measurement: PairMeasurement,
    model_shape: tuple[int, int],
    raw_shape: tuple[int, int],
) -> tuple[int, int]:
    scale_y = model_shape[0] / raw_shape[0]
    scale_x = model_shape[1] / raw_shape[1]
    return (
        int(round(measurement.shift_x * scale_x)),
        int(round(measurement.shift_y * scale_y)),
    )


def _overlap_design(
    corrected: np.ndarray,
    measurement: PairMeasurement,
    terms: Sequence[tuple[int, int]],
    raw_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = corrected.shape[-2:]
    shift_x, shift_y = _scaled_shift(
        measurement, (height, width), raw_shape
    )
    slices = aligned_slices((height, width), shift_x, shift_y)
    if slices is None:
        return None
    first_slice, second_slice = slices
    first = corrected[measurement.first][first_slice]
    second = corrected[measurement.second][second_slice]
    if min(first.shape) < 4:
        return None
    finite = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(finite) < 16:
        return None

    first_y, first_x = np.meshgrid(
        np.arange(first_slice[0].start, first_slice[0].stop),
        np.arange(first_slice[1].start, first_slice[1].stop),
        indexing="ij",
    )
    second_y, second_x = np.meshgrid(
        np.arange(second_slice[0].start, second_slice[0].stop),
        np.arange(second_slice[1].start, second_slice[1].stop),
        indexing="ij",
    )
    first_coordinates = np.column_stack(
        (
            first_y[finite] / max(height - 1, 1),
            first_x[finite] / max(width - 1, 1),
        )
    )
    second_coordinates = np.column_stack(
        (
            second_y[finite] / max(height - 1, 1),
            second_x[finite] / max(width - 1, 1),
        )
    )
    design = dct_basis(first_coordinates, terms) - dct_basis(
        second_coordinates, terms
    )
    response = np.log(np.maximum(first[finite], MIN_POSITIVE_SIGNAL))
    response -= np.log(np.maximum(second[finite], MIN_POSITIVE_SIGNAL))
    return design, response.astype(np.float64)


def overlap_chunk_mask(
    shape: tuple[int, int], axis: str, fit: bool
) -> np.ndarray:
    if axis == "horizontal":
        chunks = np.arange(shape[0])[:, None] // OVERLAP_CHUNK_SIZE
        parity = np.broadcast_to(chunks, shape) % 2
    elif axis == "vertical":
        chunks = np.arange(shape[1])[None, :] // OVERLAP_CHUNK_SIZE
        parity = np.broadcast_to(chunks, shape) % 2
    else:
        chunks = (
            np.arange(shape[0])[:, None] + np.arange(shape[1])[None, :]
        ) // OVERLAP_CHUNK_SIZE
        parity = chunks % 2
    return parity == (0 if fit else 1)


def robust_log_difference(
    first: np.ndarray, second: np.ndarray, axis: str, *, fit: bool
) -> float:
    selection = overlap_chunk_mask(first.shape, axis, fit=fit)
    local_mean = (first + second) / 2.0
    selected_mean = local_mean[selection & np.isfinite(local_mean)]
    if selected_mean.size == 0:
        raise ValueError("tile-gain overlap fit retained no finite pixels")
    low, high = np.percentile(
        selected_mean, [VALIDATION_LOW_PERCENTILE, VALIDATION_HIGH_PERCENTILE]
    )
    keep = (
        selection
        & np.isfinite(first)
        & np.isfinite(second)
        & (local_mean > low)
        & (local_mean < high)
        & (first > 0.0)
        & (second > 0.0)
    )
    if not np.any(keep):
        raise ValueError("tile-gain overlap fit retained no usable pixels")
    return float(np.median(np.log(first[keep]) - np.log(second[keep])))


def fit_tile_gains(
    corrected: np.ndarray,
    measurements: Sequence[PairMeasurement],
    raw_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, float]]:
    count = corrected.shape[0]
    rows = []
    responses = []
    for measurement in measurements:
        shift_x, shift_y = _scaled_shift(
            measurement, corrected.shape[-2:], raw_shape
        )
        slices = aligned_slices(corrected.shape[-2:], shift_x, shift_y)
        if slices is None:
            continue
        first = corrected[measurement.first][slices[0]]
        second = corrected[measurement.second][slices[1]]
        if min(first.shape) < 4:
            continue
        try:
            response = robust_log_difference(
                first, second, measurement.axis, fit=True
            )
        except ValueError:
            continue
        row = np.zeros(count, dtype=np.float64)
        row[measurement.first] = 1.0
        row[measurement.second] = -1.0
        rows.append(row)
        responses.append(response)
    if not rows:
        gains = np.ones(count, dtype=np.float32)
        return gains, {
            "gain_edges": 0,
            "gain_sd_pct": 0.0,
            "gain_min": 1.0,
            "gain_max": 1.0,
            "gain_clipped_fraction": 0.0,
        }
    design = np.stack(rows)
    response = np.asarray(responses, dtype=np.float64)
    system = design.T @ design + TILE_GAIN_RIDGE * np.eye(count)
    gains_log = np.linalg.solve(system, design.T @ response)
    gains_log -= float(np.mean(gains_log))
    cap = math.log(TILE_GAIN_MAX_FOLD)
    clipped = np.clip(gains_log, -cap, cap)
    gains = np.exp(clipped).astype(np.float32)
    return gains, {
        "gain_edges": len(rows),
        "gain_sd_pct": 100.0 * float(np.std(gains)),
        "gain_min": float(np.min(gains)),
        "gain_max": float(np.max(gains)),
        "gain_clipped_fraction": float(np.mean(clipped != gains_log)),
    }


def fit_overlap_dct(
    stack: np.ndarray,
    measurements: Sequence[PairMeasurement],
    *,
    raw_shape: tuple[int, int] | None = None,
    adaptive_tile_gains: bool = True,
) -> IlluminationModel:
    """Fit the supplied smooth overlap-DCT field, optionally with tile gains."""
    values = np.asarray(stack, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected a tile stack with shape (P, Y, X), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("flat-field training tiles must be finite")
    raw_shape = tuple(raw_shape or values.shape[-2:])
    accepted = tuple(item for item in measurements if item.accepted)
    if not accepted:
        raise ValueError("overlap-DCT requires at least one confident tile pair")

    base = fit_log_median(values)
    base_corrected = values / base[None, :, :]
    terms = [
        (order_y, order_x)
        for order_y in range(OVERLAP_DCT_ORDER + 1)
        for order_x in range(OVERLAP_DCT_ORDER + 1)
        if order_y != 0 or order_x != 0
    ]
    design_parts = []
    response_parts = []
    used_pairs = 0
    for measurement in accepted:
        part = _overlap_design(base_corrected, measurement, terms, raw_shape)
        if part is not None:
            design_parts.append(part[0])
            response_parts.append(part[1])
            used_pairs += 1
    if not design_parts:
        raise ValueError("confident pairs contained no usable overlap pixels")
    design = np.concatenate(design_parts, axis=0)
    response = np.concatenate(response_parts)
    if response.size > MAX_DCT_SAMPLES:
        selection = np.linspace(
            0, response.size - 1, MAX_DCT_SAMPLES, dtype=np.int64
        )
        design = design[selection]
        response = response[selection]
    penalty = np.asarray(
        [
            OVERLAP_DCT_RIDGE * (1.0 + order_y**2 + order_x**2) ** 2
            for order_y, order_x in terms
        ],
        dtype=np.float64,
    )
    coefficient, fit_diagnostics = robust_ridge(design, response, penalty)
    grid_y, grid_x = np.meshgrid(
        np.linspace(0.0, 1.0, values.shape[-2]),
        np.linspace(0.0, 1.0, values.shape[-1]),
        indexing="ij",
    )
    coordinates = np.column_stack((grid_y.ravel(), grid_x.ravel()))
    delta = (dct_basis(coordinates, terms) @ coefficient).reshape(values.shape[-2:])
    delta -= float(np.mean(delta))
    flat = normalize_flat(base * np.exp(delta))
    field_corrected = values / flat[None, :, :]
    if adaptive_tile_gains:
        gains, gain_diagnostics = fit_tile_gains(
            field_corrected, accepted, raw_shape
        )
    else:
        gains = np.ones(values.shape[0], dtype=np.float32)
        gain_diagnostics = {
            "gain_edges": 0,
            "gain_sd_pct": 0.0,
            "gain_min": 1.0,
            "gain_max": 1.0,
            "gain_clipped_fraction": 0.0,
        }
    diagnostics = {
        "method": "overlap_dct_gain" if adaptive_tile_gains else "overlap_dct",
        "base_method": "log_median",
        "training_tiles": int(values.shape[0]),
        "confident_pairs": len(accepted),
        "used_pairs": used_pairs,
        "dct_order": OVERLAP_DCT_ORDER,
        "dct_terms": len(terms),
        "ridge": OVERLAP_DCT_RIDGE,
        "irls_iterations": OVERLAP_DCT_IRLS_ITERATIONS,
        "dct_samples": int(response.size),
        "delta_log_range": float(np.max(delta) - np.min(delta)),
        "flat_min": float(np.min(flat)),
        "flat_max": float(np.max(flat)),
        **fit_diagnostics,
        **gain_diagnostics,
    }
    return IlluminationModel(flatfield=flat, gains=gains, diagnostics=diagnostics)
