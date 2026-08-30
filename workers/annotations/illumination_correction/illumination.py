"""Grid-aware uneven-illumination models and automatic model selection.

The implementation follows the workflow validated in the adjacent illumination
study: estimate one physical acquisition grid, fit each channel independently,
reject candidates that damage biology, and compare the remaining candidates on a
multi-metric artifact panel.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
from scipy import ndimage


MIN_PERIODS = 4
N_FOLD_BINS = 256
SEAM_SEARCH_DIVISOR = 6.0
TILE_N = 256
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


@dataclass(frozen=True)
class TileGrid:
    """Measured physical-tile lattice for one stitched mosaic."""

    pitch_y: float
    pitch_x: float
    seam_y: float
    seam_x: float
    height: int
    width: int
    seams_y: tuple[float, ...] | None = None
    seams_x: tuple[float, ...] | None = None
    seam_residual_y: float = float("nan")
    seam_residual_x: float = float("nan")
    prominence_y: float = float("nan")
    prominence_x: float = float("nan")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def is_valid(self) -> bool:
        values = (self.pitch_y, self.pitch_x, self.seam_y, self.seam_x)
        if not all(np.isfinite(v) and v > 0 for v in values[:2]):
            return False
        if not all(np.isfinite(v) for v in values[2:]):
            return False
        if not self.seams_y or not self.seams_x:
            return False
        if len(self.seams_y) < 3 or len(self.seams_x) < 3:
            return False
        pitch_ratio = max(self.pitch_y, self.pitch_x) / min(self.pitch_y, self.pitch_x)
        if pitch_ratio > 1.25:
            return False
        residuals = (
            self.seam_residual_y / max(self.pitch_y, 1e-9),
            self.seam_residual_x / max(self.pitch_x, 1e-9),
        )
        return all(np.isfinite(v) and v < 0.25 for v in residuals)

    @property
    def quality_score(self) -> float:
        """Lower is better; balances square pitch, seam residual, and peak isolation."""
        if not self.is_valid:
            return float("inf")
        pitch_disagreement = abs(math.log(self.pitch_y / self.pitch_x))
        residual = (
            self.seam_residual_y / self.pitch_y
            + self.seam_residual_x / self.pitch_x
        )
        prominence = sum(
            math.log1p(max(float(v), 0.0))
            for v in (self.prominence_y, self.prominence_x)
            if np.isfinite(v)
        )
        return float(2.0 * pitch_disagreement + residual - 0.05 * prominence)

    def _seam_array(self, axis: str) -> np.ndarray:
        seams = self.seams_y if axis == "y" else self.seams_x
        pitch = self.pitch_y if axis == "y" else self.pitch_x
        phase = self.seam_y if axis == "y" else self.seam_x
        extent = self.height if axis == "y" else self.width

        if seams:
            values = np.asarray(sorted(set(seams)), dtype=np.float64)
            gap = float(np.median(np.diff(values))) if values.size > 1 else pitch
        else:
            values = np.arange(phase % pitch, extent + pitch, pitch, dtype=np.float64)
            gap = pitch
        while values[0] > 0:
            values = np.concatenate(([values[0] - gap], values))
        while values[-1] < extent:
            values = np.concatenate((values, [values[-1] + gap]))
        return values

    def complete_seams(self, axis: str) -> np.ndarray:
        """Measured seams bounding complete physical tiles inside the image."""
        seams = self.seams_y if axis == "y" else self.seams_x
        extent = self.height if axis == "y" else self.width
        if not seams:
            raise ValueError(f"No measured {axis}-axis seams are available")
        values = np.asarray(sorted(set(seams)), dtype=np.float64)
        values = values[(values >= 0) & (values <= extent)]
        if values.size < 2:
            raise ValueError(f"Fewer than two measured {axis}-axis seams are available")
        return values

    def _fraction(self, positions: np.ndarray, axis: str) -> np.ndarray:
        seams = self._seam_array(axis)
        positions = np.asarray(positions, dtype=np.float64)
        index = np.clip(
            np.searchsorted(seams, positions, side="right") - 1,
            0,
            seams.size - 2,
        )
        widths = seams[index + 1] - seams[index]
        return ((positions - seams[index]) / np.maximum(widths, 1e-9)).astype(
            np.float32
        )

    def u_of(self, y: np.ndarray) -> np.ndarray:
        return self._fraction(y, "y")

    def v_of(self, x: np.ndarray) -> np.ndarray:
        return self._fraction(x, "x")

    def tile_of_y(self, y: np.ndarray) -> np.ndarray:
        seams = self._seam_array("y")
        return np.clip(
            np.searchsorted(seams, np.asarray(y), side="right") - 1,
            0,
            seams.size - 2,
        )

    def tile_of_x(self, x: np.ndarray) -> np.ndarray:
        seams = self._seam_array("x")
        return np.clip(
            np.searchsorted(seams, np.asarray(x), side="right") - 1,
            0,
            seams.size - 2,
        )

    def as_dict(self) -> dict:
        return {
            "pitch_y": float(self.pitch_y),
            "pitch_x": float(self.pitch_x),
            "seam_y": float(self.seam_y),
            "seam_x": float(self.seam_x),
            "seams_y": [float(v) for v in self.seams_y or ()],
            "seams_x": [float(v) for v in self.seams_x or ()],
            "seam_residual_y": float(self.seam_residual_y),
            "seam_residual_x": float(self.seam_residual_x),
            "prominence_y": float(self.prominence_y),
            "prominence_x": float(self.prominence_x),
            "quality_score": float(self.quality_score),
        }


def median_profiles(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.median(image, axis=1).astype(np.float64),
        np.median(image, axis=0).astype(np.float64),
    )


def fold(profile: np.ndarray, pitch: float, bins: int = N_FOLD_BINS) -> np.ndarray:
    labels = ((np.arange(profile.size) % pitch) / pitch * bins).astype(int) % bins
    total = np.bincount(labels, weights=profile, minlength=bins)
    count = np.bincount(labels, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def fold_amplitude(profile: np.ndarray, pitch: float, bins: int = N_FOLD_BINS) -> float:
    folded = fold(profile, pitch, bins)
    if np.all(np.isnan(folded)):
        return float("nan")
    return float(np.nanpercentile(folded, 97) - np.nanpercentile(folded, 3))


def _detrend(profile: np.ndarray, sigma: float) -> np.ndarray:
    trend = ndimage.gaussian_filter1d(profile, max(float(sigma), 1.0), mode="nearest")
    return profile / np.maximum(trend, 1e-9) - 1.0


def _periodogram_pitch(profile: np.ndarray, low: float, high: float) -> float:
    relative = _detrend(profile, profile.size / 8.0)
    n = relative.size
    power = np.abs(np.fft.rfft(relative * np.hanning(n))) ** 2
    frequencies = np.fft.rfftfreq(n)
    with np.errstate(divide="ignore"):
        pitches = 1.0 / frequencies
    keep = np.isfinite(pitches) & (pitches >= low) & (pitches <= high)
    if not keep.any():
        return float("nan")
    indices = np.nonzero(keep)[0]
    return float(pitches[indices[int(np.argmax(power[indices]))]])


def _refine_pitch(profile: np.ndarray, initial: float) -> tuple[float, float]:
    relative = _detrend(profile, initial)
    step = max(0.05, initial / 3000.0)
    candidates = np.arange(initial * 0.96, initial * 1.04 + step, step)
    amplitudes = np.asarray([fold_amplitude(relative, p) for p in candidates])
    if np.all(np.isnan(amplitudes)):
        return float(initial), float("nan")
    best = int(np.nanargmax(amplitudes))
    finite = amplitudes[np.isfinite(amplitudes)]
    prominence = float(
        amplitudes[best] / max(float(np.median(np.abs(finite))), 1e-9)
    )
    return float(candidates[best]), prominence


def _find_seams(
    profile: np.ndarray, pitch: float, phase: float
) -> tuple[np.ndarray, float]:
    relative = _detrend(profile, pitch)
    smooth = ndimage.gaussian_filter1d(
        relative, max(pitch / 40.0, 1.0), mode="nearest"
    )
    window = max(int(pitch / SEAM_SEARCH_DIVISOR), 3)
    found = []
    for predicted in np.arange(phase % pitch, profile.size, pitch):
        center = int(round(predicted))
        start = max(center - window, 0)
        stop = min(center + window + 1, profile.size)
        if stop - start < 3:
            continue
        index = start + int(np.argmin(smooth[start:stop]))
        if index <= window // 2 or index >= profile.size - 1 - window // 2:
            continue
        if not (0 < index < profile.size - 1):
            continue
        left, middle, right = smooth[index - 1 : index + 2]
        denominator = left - 2 * middle + right
        offset = (
            0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        )
        found.append(index + float(np.clip(offset, -1.0, 1.0)))

    if len(found) < 3:
        return np.asarray([], dtype=np.float64), float("inf")
    positions = np.asarray(sorted(found), dtype=np.float64)
    lattice = np.round((positions - positions[0]) / pitch)
    design = np.column_stack((lattice, np.ones_like(lattice)))
    (slope, intercept), *_ = np.linalg.lstsq(design, positions, rcond=None)
    residual = float(np.max(np.abs(positions - (slope * lattice + intercept))))
    return positions, residual


def _fit_axis(
    profile: np.ndarray, pitch_min: float, pitch_max: float
) -> tuple[float, float, np.ndarray, float, float]:
    high = min(float(pitch_max), profile.size / MIN_PERIODS)
    low = float(pitch_min)
    if high <= low:
        raise ValueError(
            f"A {profile.size}-pixel axis cannot contain {MIN_PERIODS} periods "
            f"between {low:g} and {pitch_max:g} pixels"
        )
    initial = _periodogram_pitch(profile, low, high)
    if not np.isfinite(initial):
        raise ValueError("No periodic illumination peak was found")
    refined, prominence = _refine_pitch(profile, initial)
    folded = fold(_detrend(profile, refined), refined)
    phase = float(np.nanargmin(folded)) / folded.size * refined
    seams, residual = _find_seams(profile, refined, phase)
    if seams.size >= 3:
        pitch = float(np.median(np.diff(seams)))
        phase = float(seams[0] % pitch)
    else:
        pitch = refined
    return pitch, phase, seams, residual, prominence


def fit_grid(
    image: np.ndarray, pitch_min: float = 150.0, pitch_max: float = 1400.0
) -> TileGrid:
    image = as_plane(image)
    row, column = median_profiles(image)
    py, sy, seams_y, ry, prominence_y = _fit_axis(row, pitch_min, pitch_max)
    px, sx, seams_x, rx, prominence_x = _fit_axis(column, pitch_min, pitch_max)
    return TileGrid(
        pitch_y=py,
        pitch_x=px,
        seam_y=sy,
        seam_x=sx,
        height=image.shape[0],
        width=image.shape[1],
        seams_y=tuple(float(v) for v in seams_y) or None,
        seams_x=tuple(float(v) for v in seams_x) or None,
        seam_residual_y=ry,
        seam_residual_x=rx,
        prominence_y=prominence_y,
        prominence_x=prominence_x,
    )


def as_plane(image: np.ndarray) -> np.ndarray:
    plane = np.asarray(image).squeeze()
    if plane.ndim != 2:
        raise ValueError(f"Expected a 2-D image plane, got shape {plane.shape}")
    return plane


def _frame_image(
    tile_client, dataset_id: str, coordinates: dict, channel: int
) -> np.ndarray:
    frame = tile_client.coordinatesToFrameIndex(
        coordinates["XY"], coordinates["Z"], coordinates["Time"], channel
    )
    return as_plane(tile_client.getRegion(dataset_id, frame=frame))


def _available_channels(tile_client) -> list[int]:
    frames = tile_client.tiles.get("frames", [])
    channels = sorted({int(frame.get("IndexC", 0)) for frame in frames})
    if channels:
        return channels
    count = int(tile_client.tiles.get("IndexRange", {}).get("IndexC", 1))
    return list(range(max(count, 1)))


def choose_reference_grid(
    tile_client,
    dataset_id: str,
    coordinates: dict,
    mode: str,
    reference_channel: int,
    pitch_min: float,
    pitch_max: float,
    progress: Callable[[float, str, str], None] | None = None,
) -> tuple[TileGrid, int, list[dict]]:
    """Return the best fitted grid from the dominant cross-channel cluster."""
    automatic = mode == "Automatically choose best channel"
    channels = (
        _available_channels(tile_client) if automatic else [int(reference_channel)]
    )
    reports: list[dict] = []
    fitted: list[tuple[int, TileGrid, dict]] = []

    for index, channel in enumerate(channels):
        if progress:
            progress(
                index / max(len(channels), 1),
                "Illumination correction",
                f"Evaluating channel {channel + 1} as the grid reference",
            )
        try:
            image = _frame_image(tile_client, dataset_id, coordinates, channel)
            grid = fit_grid(image, pitch_min=pitch_min, pitch_max=pitch_max)
            report = {
                "channel": channel,
                "valid": bool(grid.is_valid),
                **grid.as_dict(),
            }
            if grid.is_valid:
                fitted.append((channel, grid, report))
        except Exception as exc:
            report = {"channel": channel, "valid": False, "error": str(exc)}
        reports.append(report)

    if not fitted:
        detail = "; ".join(
            f"channel {r['channel'] + 1}: {r.get('error', 'low-confidence grid')}"
            for r in reports
        )
        raise ValueError(
            "No channel produced a reliable physical tile grid. "
            f"Check the reference plane and tile-pitch range. {detail}"
        )

    # A real acquisition grid is shared across channels. Prefer a grid supported by
    # the most other channels, then the best individual residual/prominence score.
    for channel, grid, report in fitted:
        agreement = sum(
            abs(other.pitch_y / grid.pitch_y - 1.0) <= 0.075
            and abs(other.pitch_x / grid.pitch_x - 1.0) <= 0.075
            for _, other, _ in fitted
        )
        report["cross_channel_agreement"] = int(agreement)

    channel, grid, _ = min(
        fitted,
        key=lambda item: (
            -item[2]["cross_channel_agreement"],
            item[1].quality_score,
            item[0],
        ),
    )
    return grid, channel, reports


def normalize_flat(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    if not np.isfinite(field).all() or np.any(field <= 0):
        raise ValueError("A fitted flatfield must be finite and strictly positive")
    mean = float(np.mean(field))
    if not np.isfinite(mean) or mean < 1e-12:
        raise ValueError("A fitted flatfield must have a positive finite mean")
    return (field / mean).astype(np.float32)


def tile_stack(image: np.ndarray, grid: TileGrid, n: int = TILE_N) -> np.ndarray:
    """Resample complete, seam-to-seam physical tiles into normalized coordinates."""
    sy = grid.complete_seams("y")
    sx = grid.complete_seams("x")
    fraction = np.linspace(0.0, 1.0, int(n), endpoint=False, dtype=np.float32)
    source = np.asarray(image, dtype=np.float32)
    output = []
    for y0, y1 in zip(sy[:-1], sy[1:]):
        ys = y0 + fraction * (y1 - y0)
        for x0, x1 in zip(sx[:-1], sx[1:]):
            xs = x0 + fraction * (x1 - x0)
            coordinates = np.stack(np.meshgrid(ys, xs, indexing="ij"))
            output.append(
                ndimage.map_coordinates(source, coordinates, order=1, mode="nearest")
            )
    if len(output) < 4:
        raise ValueError("At least four complete physical tiles are required")
    return np.stack(output).astype(np.float32)


def expand_tile_field(field: np.ndarray, grid: TileGrid) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"Expected a 2-D tile field, got {field.shape}")
    u = grid.u_of(np.arange(grid.height))
    v = grid.v_of(np.arange(grid.width))
    iy = np.clip((u * field.shape[0]).astype(np.int32), 0, field.shape[0] - 1)
    ix = np.clip((v * field.shape[1]).astype(np.int32), 0, field.shape[1] - 1)
    return field[np.ix_(iy, ix)].astype(np.float32)


def smooth_periodic(field: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(field, sigma, mode="wrap")


def _tile_gain_map(gains: np.ndarray, grid: TileGrid) -> np.ndarray:
    sy = grid.complete_seams("y")
    sx = grid.complete_seams("x")
    values = np.asarray(gains, dtype=np.float32).reshape(len(sy) - 1, len(sx) - 1)
    iy = np.clip(
        np.searchsorted(sy, np.arange(grid.height), side="right") - 1,
        0,
        values.shape[0] - 1,
    )
    ix = np.clip(
        np.searchsorted(sx, np.arange(grid.width), side="right") - 1,
        0,
        values.shape[1] - 1,
    )
    return values[np.ix_(iy, ix)].astype(np.float32)


def _estimate_tile_gains(corrected_stack: np.ndarray) -> np.ndarray:
    levels = np.percentile(corrected_stack, 25.0, axis=(1, 2))
    median = max(float(np.median(levels)), 1e-9)
    gains = np.clip(levels / median, 0.5, 2.0)
    return (gains / max(float(np.mean(gains)), 1e-9)).astype(np.float32)


class PeriodicFieldModel:
    """Compact normalized-tile field expanded only while applying a plane."""

    def __init__(
        self,
        name: str,
        grid: TileGrid,
        flat_tile: np.ndarray,
        dark: float | np.ndarray = 0.0,
        gains: np.ndarray | None = None,
        diagnostics: dict | None = None,
    ):
        self.name = name
        self.grid = grid
        self.flat_tile = normalize_flat(flat_tile)
        dark_values = np.asarray(dark)
        if not np.isfinite(dark_values).all():
            raise ValueError("A fitted darkfield must be finite")
        self.dark = dark
        if gains is not None:
            gains = np.asarray(gains, dtype=np.float32)
            if not np.isfinite(gains).all() or np.any(gains <= 0):
                raise ValueError("Fitted per-tile gains must be finite and positive")
        self.gains = gains
        self.diagnostics = diagnostics or {}

    def apply(self, image: np.ndarray) -> np.ndarray:
        raw = as_plane(image)
        if raw.shape != self.grid.shape:
            raise ValueError(
                f"Model expects planes of shape {self.grid.shape}, got {raw.shape}"
            )
        flat = expand_tile_field(self.flat_tile, self.grid)
        if self.gains is not None:
            flat *= _tile_gain_map(self.gains, self.grid)
        flat = np.maximum(flat, 1e-3)

        output = raw.astype(np.float32, copy=True)
        if np.ndim(self.dark) == 0:
            dark_mean = float(self.dark)
            output -= dark_mean
        else:
            dark_map = expand_tile_field(np.asarray(self.dark), self.grid)
            dark_mean = float(np.mean(self.dark))
            output -= dark_map
        output /= flat
        output += dark_mean
        return output.astype(np.float32)


class IdentityModel:
    """Explicit no-correction baseline used by automatic selection."""

    def __init__(self, grid: TileGrid, diagnostics: dict | None = None):
        self.name = "identity"
        self.grid = grid
        self.diagnostics = diagnostics or {}

    def apply(self, image: np.ndarray) -> np.ndarray:
        raw = as_plane(image)
        if raw.shape != self.grid.shape:
            raise ValueError(
                f"Model expects planes of shape {self.grid.shape}, got {raw.shape}"
            )
        return raw.astype(np.float32, copy=True)


class SplitHalfAffineModel:
    def __init__(
        self,
        grid: TileGrid,
        gain_y: np.ndarray,
        gain_x: np.ndarray,
        offset_y: np.ndarray,
        offset_x: np.ndarray,
        diagnostics: dict,
    ):
        self.name = "split_half_affine"
        self.grid = grid
        self.gain_y = gain_y.astype(np.float32)
        self.gain_x = gain_x.astype(np.float32)
        self.offset_y = offset_y.astype(np.float32)
        self.offset_x = offset_x.astype(np.float32)
        self.diagnostics = diagnostics

    def apply(self, image: np.ndarray) -> np.ndarray:
        raw = as_plane(image)
        if raw.shape != self.grid.shape:
            raise ValueError(
                f"Model expects planes of shape {self.grid.shape}, got {raw.shape}"
            )
        flat = expand_tile_field(np.outer(self.gain_y, self.gain_x), self.grid)
        offset = expand_tile_field(
            self.offset_y[:, None] + self.offset_x[None, :], self.grid
        )
        return (raw.astype(np.float32) / np.maximum(flat, 1e-3) - offset).astype(
            np.float32
        )


def fit_basic(
    raw: np.ndarray,
    grid: TileGrid,
    *,
    darkfield: bool,
    per_tile_gain: bool = True,
    tile_n: int = TILE_N,
) -> PeriodicFieldModel:
    """Fit BaSiCPy to the normalized stack of complete physical tiles."""
    try:
        from basicpy import BaSiC
    except ImportError as exc:
        raise RuntimeError("BaSiCPy is not installed in the worker image") from exc

    stack = tile_stack(raw, grid, n=tile_n)
    estimator = BaSiC(
        get_darkfield=bool(darkfield),
        sparse_cost_darkfield=0.01,
        fitting_mode="approximate",
        sort_intensity=False,
        working_size=128,
        max_iterations=500,
    )
    estimator.fit(stack.astype(np.float32))
    flat_tile = np.asarray(estimator.flatfield, dtype=np.float32).squeeze()
    if flat_tile.shape != (tile_n, tile_n):
        raise ValueError(
            f"BaSiC returned an unexpected flatfield shape {flat_tile.shape}"
        )

    if darkfield:
        dark_tile = np.asarray(estimator.darkfield, dtype=np.float32).squeeze()
        if dark_tile.ndim == 0 or dark_tile.size == 1:
            dark_tile = np.full_like(flat_tile, float(dark_tile))
    else:
        dark_tile = np.zeros_like(flat_tile)
    if dark_tile.shape != flat_tile.shape:
        raise ValueError(
            f"BaSiC returned an unexpected darkfield shape {dark_tile.shape}"
        )

    flat_tile = normalize_flat(flat_tile)
    gains = None
    if per_tile_gain:
        corrected_stack = (stack - dark_tile[None]) / np.maximum(flat_tile[None], 1e-3)
        gains = _estimate_tile_gains(corrected_stack)

    name = "basic_darkfield_on" if darkfield else "basic_darkfield_off"
    diagnostics = {
        "darkfield": bool(darkfield),
        "per_tile_gain": bool(per_tile_gain),
        "tile_n": int(tile_n),
        "n_tiles": int(stack.shape[0]),
        "darkfield_mean": float(np.mean(dark_tile)),
        "darkfield_range": [float(np.min(dark_tile)), float(np.max(dark_tile))],
        "flatfield_range": [float(np.min(flat_tile)), float(np.max(flat_tile))],
        "tile_gain_sd_pct": (
            float(100.0 * np.std(gains)) if gains is not None else 0.0
        ),
    }
    return PeriodicFieldModel(
        name,
        grid,
        flat_tile,
        dark=dark_tile,
        gains=gains,
        diagnostics=diagnostics,
    )


def periodic_poisson(gy: np.ndarray, gx: np.ndarray) -> np.ndarray:
    rows, columns = gy.shape
    ky = np.fft.fftfreq(rows)[:, None]
    kx = np.fft.fftfreq(columns)[None, :]
    my = np.exp(2j * np.pi * ky) - 1.0
    mx = np.exp(2j * np.pi * kx) - 1.0
    denominator = np.abs(my) ** 2 + np.abs(mx) ** 2
    denominator[0, 0] = 1.0
    numerator = np.conj(my) * np.fft.fft2(gy) + np.conj(mx) * np.fft.fft2(gx)
    potential = np.fft.ifft2(numerator / denominator).real
    return potential - potential.mean()


def fit_log_gradient(
    raw: np.ndarray,
    grid: TileGrid,
    *,
    n: int = TILE_N,
    smooth_sigma: float = 2.0,
    field_smooth: float = 1.5,
    per_tile_gain: bool = True,
) -> PeriodicFieldModel:
    source = np.asarray(raw)
    array = as_plane(source).astype(np.float32)
    positive = array[array > 0]
    if positive.size == 0:
        raise ValueError("Folded log-gradient requires positive image intensities")
    floor = max(float(np.percentile(positive, 1)) * 0.05, 1.0)
    log_image = np.log(np.maximum(array, floor))
    if smooth_sigma:
        log_image = ndimage.gaussian_filter(log_image, float(smooth_sigma))

    gy = np.empty_like(log_image)
    gx = np.empty_like(log_image)
    gy[:-1] = np.diff(log_image, axis=0)
    gy[-1] = gy[-2]
    gx[:, :-1] = np.diff(log_image, axis=1)
    gx[:, -1] = gx[:, -2]

    if np.issubdtype(source.dtype, np.integer):
        saturation = float(np.iinfo(source.dtype).max)
        bad = array >= saturation
        if bad.any():
            bad = ndimage.binary_dilation(bad, np.ones((3, 3), dtype=bool))
            gy = np.where(bad, np.nan, gy)
            gx = np.where(bad, np.nan, gx)

    with np.errstate(invalid="ignore"):
        folded_gy = np.nanmedian(tile_stack(gy, grid, n=n), axis=0)
        folded_gx = np.nanmedian(tile_stack(gx, grid, n=n), axis=0)
    folded_gy = np.nan_to_num(folded_gy) * (grid.pitch_y / n)
    folded_gx = np.nan_to_num(folded_gx) * (grid.pitch_x / n)

    log_field = periodic_poisson(folded_gy, folded_gx)
    if field_smooth:
        log_field = smooth_periodic(log_field, float(field_smooth))
    flat_tile = normalize_flat(np.exp(log_field).astype(np.float32))

    gains = None
    if per_tile_gain:
        stack = tile_stack(array, grid, n=128)
        sampled_flat = ndimage.zoom(
            flat_tile,
            (128 / flat_tile.shape[0], 128 / flat_tile.shape[1]),
            order=1,
            mode="wrap",
        )
        sampled_flat = sampled_flat[:128, :128]
        gains = _estimate_tile_gains(stack / np.maximum(sampled_flat[None], 1e-3))

    diagnostics = {
        "per_tile_gain": bool(per_tile_gain),
        "tile_n": int(n),
        "log_field_range": float(np.exp(log_field.max() - log_field.min())),
        "tile_gain_sd_pct": (
            float(100.0 * np.std(gains)) if gains is not None else 0.0
        ),
    }
    return PeriodicFieldModel(
        "fold_log_gradient",
        grid,
        flat_tile,
        dark=0.0,
        gains=gains,
        diagnostics=diagnostics,
    )


LOCATION_QUANTILES = (0.10, 0.25, 0.50)
PROFILE_QUANTILES = (0.10, 0.25, 0.50, 0.75)
BASE_SCALES = np.asarray((3.0, 9.0, 27.0, 81.0, 243.0))


@dataclass
class _Profiles:
    location_y: np.ndarray
    location_x: np.ndarray
    spread_y: np.ndarray
    spread_x: np.ndarray


def _extract_profiles(
    stack: np.ndarray,
    gain_y: np.ndarray | None = None,
    gain_x: np.ndarray | None = None,
) -> _Profiles:
    values = np.asarray(stack, dtype=np.float32).copy()
    if gain_y is not None and gain_x is not None:
        values /= np.maximum(gain_y[None, :, None] * gain_x[None, None, :], 1e-6)
    values -= np.median(values, axis=(1, 2), keepdims=True)
    yq = np.quantile(values, PROFILE_QUANTILES, axis=2).transpose(1, 0, 2)
    xq = np.quantile(values, PROFILE_QUANTILES, axis=1).transpose(1, 0, 2)
    return _Profiles(
        location_y=yq[:, : len(LOCATION_QUANTILES)],
        location_x=xq[:, : len(LOCATION_QUANTILES)],
        spread_y=yq[:, 3] - yq[:, 1],
        spread_x=xq[:, 3] - xq[:, 1],
    )


def _aggregate_location(
    profiles: np.ndarray, indices: np.ndarray | slice = slice(None)
) -> np.ndarray:
    quantiles = np.median(profiles[indices], axis=0)
    quantiles -= np.mean(quantiles, axis=1, keepdims=True)
    result = np.median(quantiles, axis=0)
    return result - np.mean(result)


def _aggregate_log_spread(
    profiles: np.ndarray, indices: np.ndarray | slice = slice(None)
) -> np.ndarray:
    spread = np.maximum(np.median(profiles[indices], axis=0), 1e-6)
    result = np.log(spread)
    return result - np.mean(result)


def _split_indices(count: int, split_count: int, seed: int):
    if count < 4:
        raise ValueError("At least four tile cells are required")
    rng = np.random.default_rng(seed)
    first_size = (count + 1) // 2
    return [
        (order[:first_size], order[first_size:])
        for order in (rng.permutation(count) for _ in range(split_count))
    ]


def _concordance(left: np.ndarray, right: np.ndarray) -> float:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = np.mean(left * left) + np.mean(right * right)
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.clip(2.0 * np.mean(left * right) / denominator, 0.0, 1.0))


def _multiscale_bands(curve: np.ndarray, scales: np.ndarray) -> list[np.ndarray]:
    smoothed = [ndimage.gaussian_filter1d(curve, s, mode="wrap") for s in scales]
    return [
        smoothed[-1],
        *[
            smoothed[index] - smoothed[index + 1]
            for index in range(len(smoothed) - 2, -1, -1)
        ],
    ]


def _denoise_reproducible_curve(
    profiles: np.ndarray,
    aggregate: Callable,
    splits,
    scales: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    full = _multiscale_bands(aggregate(profiles), scales)
    halves = [
        (
            _multiscale_bands(aggregate(profiles, left), scales),
            _multiscale_bands(aggregate(profiles, right), scales),
        )
        for left, right in splits
    ]
    result = np.zeros_like(full[0])
    diagnostics = []
    for index, band in enumerate(full):
        reliability = float(
            np.median([_concordance(a[index], b[index]) for a, b in halves])
        )
        if reliability < 0.05:
            reliability = 0.0
        shrinkage = 2 * reliability / (1 + reliability) if reliability else 0.0
        result += shrinkage * band
        diagnostics.append(
            {
                "half_sample_reliability": reliability,
                "full_sample_shrinkage": shrinkage,
            }
        )
    return result - np.mean(result), diagnostics


def fit_split_half_affine(
    raw: np.ndarray,
    grid: TileGrid,
    *,
    profile_size: int = TILE_N,
    split_count: int = 16,
    seed: int = 2026,
    max_gain: float = 2.0,
) -> SplitHalfAffineModel:
    stack = tile_stack(raw, grid, n=profile_size)
    splits = _split_indices(stack.shape[0], split_count, seed)
    scales = np.maximum(0.5, BASE_SCALES * profile_size / 1024.0)
    profiles = _extract_profiles(stack)

    log_gain_y, gain_y_diag = _denoise_reproducible_curve(
        profiles.spread_y, _aggregate_log_spread, splits, scales
    )
    log_gain_x, gain_x_diag = _denoise_reproducible_curve(
        profiles.spread_x, _aggregate_log_spread, splits, scales
    )
    limit = math.log(max_gain)
    gain_y = np.exp(np.clip(log_gain_y, -limit / 2, limit / 2))
    gain_x = np.exp(np.clip(log_gain_x, -limit / 2, limit / 2))
    normalization = math.sqrt(float(np.mean(np.outer(gain_y, gain_x))))
    gain_y /= max(normalization, 1e-9)
    gain_x /= max(normalization, 1e-9)

    corrected_profiles = _extract_profiles(stack, gain_y, gain_x)
    offset_y, offset_y_diag = _denoise_reproducible_curve(
        corrected_profiles.location_y, _aggregate_location, splits, scales
    )
    offset_x, offset_x_diag = _denoise_reproducible_curve(
        corrected_profiles.location_x, _aggregate_location, splits, scales
    )
    center = float(np.median(offset_y[:, None] + offset_x[None, :]))
    offset_y -= center / 2
    offset_x -= center / 2

    diagnostics = {
        "profile_size": int(profile_size),
        "split_count": int(split_count),
        "seed": int(seed),
        "gain_range": [
            float(np.min(np.outer(gain_y, gain_x))),
            float(np.max(np.outer(gain_y, gain_x))),
        ],
        "offset_range": [
            float(np.min(offset_y[:, None] + offset_x[None, :])),
            float(np.max(offset_y[:, None] + offset_x[None, :])),
        ],
        "gain_y_bands": gain_y_diag,
        "gain_x_bands": gain_x_diag,
        "offset_y_bands": offset_y_diag,
        "offset_x_bands": offset_x_diag,
    }
    return SplitHalfAffineModel(
        grid, gain_y, gain_x, offset_y, offset_x, diagnostics
    )


# Metric constants are fixed to the values used by the illumination study.
BG_PCTL = 10
UV_POLY_DEGREE = 5
N_HARMONICS = 4
PSF_SIGMA = 1.3
SPOT_K = 5.0
MIN_SPOT_COUNT = 20
HF_CUTOFF = 0.10
HF_CROPS = 4
HF_CROP = 2048
HF_LOCAL_SIGMA = 32.0
MAD_TO_SIGMA = 1.4826
A1_BINS = 32


def _metric_block(grid: TileGrid, preferred: int = 64) -> int:
    return max(4, min(preferred, int(min(grid.pitch_y, grid.pitch_x) / 6)))


def block_percentile(
    image: np.ndarray, block: int, percentile: float = BG_PCTL
) -> np.ndarray:
    height, width = image.shape
    block = max(2, min(int(block), height, width))
    hh, ww = (height // block) * block, (width // block) * block
    if hh == 0 or ww == 0:
        raise ValueError("Image is too small for block-percentile metrics")
    cells = image[:hh, :ww].reshape(hh // block, block, ww // block, block)
    return np.percentile(cells.astype(np.float32), percentile, axis=(1, 3))


def _poly_basis(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack(
        [
            (u**du) * (v**dv)
            for du in range(degree + 1)
            for dv in range(degree + 1 - du)
        ]
    )


def a1_fold_amplitude(image: np.ndarray, grid: TileGrid) -> dict:
    row, column = median_profiles(image)
    scale = max(abs(float(np.median(image))), 1e-6)
    amplitudes = {}
    for name, profile, coordinate in (
        ("y", row, grid.u_of(np.arange(row.size))),
        ("x", column, grid.v_of(np.arange(column.size))),
    ):
        labels = np.clip((coordinate * A1_BINS).astype(int), 0, A1_BINS - 1)
        values = np.asarray(
            [
                np.median(profile[labels == index])
                if np.any(labels == index)
                else np.nan
                for index in range(A1_BINS)
            ]
        )
        amplitudes[name] = float(np.nanmax(values) - np.nanmin(values))
    return {
        "A1_fold_amp_y": amplitudes["y"],
        "A1_fold_amp_x": amplitudes["x"],
        "A1_fold_amp_rel_pct": 100 * max(amplitudes.values()) / scale,
    }


def a2_harmonic_power(image: np.ndarray, grid: TileGrid) -> dict:
    row, column = median_profiles(image)
    output = {}
    for name, profile, pitch in (
        ("y", row, grid.pitch_y),
        ("x", column, grid.pitch_x),
    ):
        trend = ndimage.gaussian_filter1d(profile, pitch, mode="nearest")
        relative = profile / np.maximum(trend, 1e-6) - 1.0
        spectrum = np.fft.rfft(relative)
        bins: list[int] = []
        for harmonic in range(1, N_HARMONICS + 1):
            center = int(round(harmonic * relative.size / pitch))
            if center < spectrum.size - 2:
                bins.extend(range(max(center - 2, 1), min(center + 3, spectrum.size)))
        bins = sorted(set(bins))
        if not bins:
            output[f"A2_harmonic_mod_pct_{name}"] = float("nan")
            continue
        selected = np.zeros_like(spectrum)
        selected[bins] = spectrum[bins]
        output[f"A2_harmonic_mod_pct_{name}"] = float(
            100 * np.std(np.fft.irfft(selected, n=relative.size))
        )
    values = [v for v in output.values() if np.isfinite(v)]
    output["A2_harmonic_mod_pct_max"] = max(values) if values else float("nan")
    return output


def _block_coordinates(grid: TileGrid, shape: tuple[int, int], block: int):
    y = (np.arange(shape[0]) + 0.5) * block
    x = (np.arange(shape[1]) + 0.5) * block
    return grid.u_of(y), grid.v_of(x), grid.tile_of_y(y), grid.tile_of_x(x)


def _blocks_spanning_seams(grid: TileGrid, shape: tuple[int, int], block: int):
    y = np.arange(shape[0]) * block
    x = np.arange(shape[1]) * block
    return (
        grid.tile_of_y(y) != grid.tile_of_y(y + block - 1),
        grid.tile_of_x(x) != grid.tile_of_x(x + block - 1),
    )


def a3_uv_dependence(image: np.ndarray, grid: TileGrid) -> dict:
    block = _metric_block(grid)
    background = block_percentile(image, block)
    u, v, tile_y, tile_x = _block_coordinates(grid, background.shape, block)
    span_y, span_x = _blocks_spanning_seams(grid, background.shape, block)
    keep = ~(span_y[:, None] | span_x[None, :])
    U = np.broadcast_to(u[:, None], background.shape)[keep]
    V = np.broadcast_to(v[None, :], background.shape)[keep]
    parity = (np.add.outer(tile_y, tile_x) % 2).astype(bool)[keep]
    values = np.log(np.maximum(background, 1e-6))[keep]
    basis = _poly_basis(U, V, UV_POLY_DEGREE)
    train = ~parity
    if train.sum() < basis.shape[1] * 3 or parity.sum() < 10:
        return {
            "A3_uv_var_explained": float("nan"),
            "A3_uv_modulation_pct": float("nan"),
        }
    coefficients, *_ = np.linalg.lstsq(basis[train], values[train], rcond=None)
    predicted = basis @ coefficients
    residual = values[parity] - predicted[parity]
    variance = float(np.var(values[parity]))
    return {
        "A3_uv_var_explained": float(
            1 - np.var(residual) / max(variance, 1e-12)
        ),
        "A3_uv_modulation_pct": float(100 * np.std(predicted[parity])),
    }


def a5_background_range(image: np.ndarray, grid: TileGrid) -> dict:
    background = block_percentile(image, _metric_block(grid))
    low, high = np.percentile(background, [1, 99])
    return {
        "A5_bg_range_ratio": float(high / max(low, 1e-6)),
        "A5_bg_p1": float(low),
        "A5_bg_p99": float(high),
    }


def a6_tile_level(image: np.ndarray, grid: TileGrid) -> dict:
    block = _metric_block(grid, preferred=32)
    background = block_percentile(image, block)
    u, v, tile_y, tile_x = _block_coordinates(grid, background.shape, block)
    span_y, span_x = _blocks_spanning_seams(grid, background.shape, block)
    keep_y = (u > 0.2) & (u < 0.8) & ~span_y
    keep_x = (v > 0.2) & (v < 0.8) & ~span_x
    levels = {}
    if keep_y.sum() >= 2 and keep_x.sum() >= 2:
        subset = background[np.ix_(keep_y, keep_x)]
        iy, ix = tile_y[keep_y], tile_x[keep_x]
        for y_index in np.unique(iy):
            for x_index in np.unique(ix):
                cell = subset[np.ix_(iy == y_index, ix == x_index)]
                if cell.size >= 4:
                    levels[(int(y_index), int(x_index))] = float(np.median(cell))
    if len(levels) < 10:
        return {"A6_tile_level_sd_pct": float("nan"), "A6_n_tiles": len(levels)}

    keys = np.asarray(list(levels))
    values = np.asarray([levels[tuple(key)] for key in keys])
    if np.any(values <= 0):
        return {"A6_tile_level_sd_pct": float("nan"), "A6_n_tiles": len(levels)}
    y = keys[:, 0].astype(float)
    x = keys[:, 1].astype(float)
    y = (y - y.mean()) / max(y.std(), 1e-9)
    x = (x - x.mean()) / max(x.std(), 1e-9)
    basis = np.column_stack(
        [y**a * x**b for a in range(3) for b in range(3 - a)]
    )
    log_values = np.log(values)
    coefficients, *_ = np.linalg.lstsq(basis, log_values, rcond=None)
    residual = log_values - basis @ coefficients
    return {
        "A6_tile_level_sd_pct": float(100 * np.std(residual)),
        "A6_n_tiles": int(values.size),
    }


def _robust_sigma(values: np.ndarray) -> float:
    return float(
        MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))) + 1e-12
    )


def p1_spots(image: np.ndarray, grid: TileGrid) -> dict:
    values = np.asarray(image, dtype=np.float32)
    response = -ndimage.gaussian_laplace(values, PSF_SIGMA)
    threshold = SPOT_K * _robust_sigma(response[::4, ::4])
    peaks = (response >= threshold) & (
        ndimage.maximum_filter(response, size=3) == response
    )
    y, x = np.nonzero(peaks)
    if y.size < MIN_SPOT_COUNT:
        return {
            "P1_spot_count": int(y.size),
            "P1_spot_uniformity": float("nan"),
            "P1_applicable": False,
        }
    u, v = grid.u_of(y), grid.v_of(x)
    inner = (np.abs(u - 0.5) < 0.25) & (np.abs(v - 0.5) < 0.25)
    outer = (np.abs(u - 0.5) > 0.40) | (np.abs(v - 0.5) > 0.40)
    # Jeffreys-style pseudocounts keep the ratio finite and symmetric when
    # either region has no detections. A zero outer count must not look like an
    # unavailable metric while a large outer count is penalized.
    density_inner = (inner.sum() + 0.5) / 0.25
    density_outer = (outer.sum() + 0.5) / 0.36
    return {
        "P1_spot_count": int(y.size),
        "P1_spot_uniformity": float(density_outer / max(density_inner, 1e-9)),
        "P1_applicable": True,
    }


def build_object_mask(
    raw: np.ndarray, max_objects: int = 2000, min_area: int = 50
) -> tuple[np.ndarray, int]:
    values = np.asarray(raw, dtype=np.float32)
    best = (np.zeros(values.shape, dtype=np.int32), 0)
    for percentile in (99.5, 99.0, 98.0, 95.0, 90.0):
        labels, count = ndimage.label(values > np.percentile(values, percentile))
        if count == 0:
            continue
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        keep = np.nonzero(areas >= min_area)[0]
        if keep.size > max_objects:
            keep = keep[np.argsort(areas[keep])[::-1][:max_objects]]
        remap = np.zeros(areas.size, dtype=np.int32)
        remap[keep] = np.arange(1, keep.size + 1)
        best = remap[labels], int(keep.size)
        if keep.size >= 10:
            break
    return best


def p2_object_intensity(
    raw: np.ndarray,
    corrected: np.ndarray,
    labels: np.ndarray,
    count: int,
) -> dict:
    if count < 10:
        return {
            "P2_n_objects": count,
            "P2_spearman": float("nan"),
            "P2_applicable": False,
        }
    indices = np.arange(1, count + 1)
    raw_sum = ndimage.sum_labels(raw.astype(np.float64), labels, indices)
    corrected_sum = ndimage.sum_labels(
        corrected.astype(np.float64), labels, indices
    )
    keep = (raw_sum > 0) & (corrected_sum > 0)
    if keep.sum() < 10:
        return {
            "P2_n_objects": int(keep.sum()),
            "P2_spearman": float("nan"),
            "P2_applicable": False,
        }
    from scipy.stats import spearmanr

    return {
        "P2_n_objects": int(keep.sum()),
        "P2_spearman": float(spearmanr(raw_sum[keep], corrected_sum[keep]).statistic),
        "P2_applicable": True,
    }


def _local_normalize(image: np.ndarray) -> np.ndarray:
    smooth = ndimage.gaussian_filter(image, HF_LOCAL_SIGMA, mode="nearest")
    return image / np.maximum(smooth, 1e-6)


def p3_high_frequency(raw: np.ndarray, corrected: np.ndarray) -> dict:
    height, width = raw.shape
    crop = min(HF_CROP, height, width)
    if crop < 8:
        return {
            "P3_hf_power_ratio": float("nan"),
            "P3_applicable": False,
            "P3_n_crops": 0,
        }
    fy = np.fft.fftfreq(crop)
    fx = np.fft.rfftfreq(crop)
    mask = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) > HF_CUTOFF
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(HF_CROPS):
        y = int(rng.integers(0, max(height - crop + 1, 1)))
        x = int(rng.integers(0, max(width - crop + 1, 1)))
        before = _local_normalize(
            raw[y : y + crop, x : x + crop].astype(np.float32)
        )
        after = _local_normalize(
            corrected[y : y + crop, x : x + crop].astype(np.float32)
        )
        power_before = float((np.abs(np.fft.rfft2(before)) ** 2)[mask].sum())
        power_after = float((np.abs(np.fft.rfft2(after)) ** 2)[mask].sum())
        if np.isfinite(power_before) and power_before > 1e-20:
            ratios.append(power_after / power_before)
    if not ratios:
        return {
            "P3_hf_power_ratio": float("nan"),
            "P3_applicable": False,
            "P3_n_crops": 0,
        }
    return {
        "P3_hf_power_ratio": float(np.mean(ratios)),
        "P3_applicable": True,
        "P3_n_crops": len(ratios),
    }


def p5_range(raw: np.ndarray, corrected: np.ndarray) -> dict:
    raw_values = np.asarray(raw)
    corrected_values = np.asarray(corrected)
    finite = np.isfinite(corrected_values)
    source_finite = np.isfinite(raw_values)
    finite_values = corrected_values[finite]
    return {
        "P5_frac_source_nonfinite": float(
            np.count_nonzero(~source_finite) / raw_values.size
        ),
        "P5_frac_nonfinite": float(
            np.count_nonzero(~finite) / corrected_values.size
        ),
        "P5_frac_nonpositive": float(
            np.count_nonzero(finite & (corrected_values <= 0))
            / corrected_values.size
        ),
        "P5_frac_new_nonpositive": float(
            np.count_nonzero(
                source_finite
                & (raw_values > 0)
                & finite
                & (corrected_values <= 0)
            )
            / corrected_values.size
        ),
        "P5_min": (
            float(np.min(finite_values)) if finite_values.size else float("nan")
        ),
        "P5_max": (
            float(np.max(finite_values)) if finite_values.size else float("nan")
        ),
    }


GUARDRAILS = {
    "P2_spearman": ("min", 0.98),
    "P3_hf_power_ratio": ("min", 0.90),
    "P5_frac_source_nonfinite": ("max", 0.0),
    "P5_frac_nonfinite": ("max", 0.0),
    "P5_frac_new_nonpositive": ("max", 1e-4),
}
CONDITIONAL_GUARDRAILS = {
    "P2_spearman": "P2_applicable",
    "P3_hf_power_ratio": "P3_applicable",
}


def check_guardrails(metrics: dict) -> list[str]:
    violations = []
    for key, (kind, limit) in GUARDRAILS.items():
        applicable_key = CONDITIONAL_GUARDRAILS.get(key)
        if applicable_key is not None and metrics.get(applicable_key) is False:
            continue
        value = metrics.get(key)
        if value is None or not np.isfinite(value):
            violations.append(f"{key} is unavailable or non-finite")
            continue
        if kind == "min" and value < limit:
            violations.append(f"{key}={value:.4g} < {limit}")
        if kind == "max" and value > limit:
            violations.append(f"{key}={value:.4g} > {limit}")
    return violations


def unavailable_guardrails(metrics: dict) -> list[str]:
    return [
        key
        for key, applicable_key in CONDITIONAL_GUARDRAILS.items()
        if metrics.get(applicable_key) is False
    ]


def preservation_metrics(
    raw: np.ndarray,
    corrected: np.ndarray,
    labels: np.ndarray | None = None,
    count: int = 0,
) -> dict:
    if labels is None:
        labels, count = build_object_mask(raw)
    metrics = {}
    metrics.update(p2_object_intensity(raw, corrected, labels, count))
    metrics.update(p3_high_frequency(raw, corrected))
    metrics.update(p5_range(raw, corrected))
    metrics["guardrail_violations"] = check_guardrails(metrics)
    metrics["guardrail_unavailable"] = unavailable_guardrails(metrics)
    return metrics


def evaluate(
    corrected: np.ndarray,
    raw: np.ndarray,
    grid: TileGrid,
    labels: np.ndarray | None = None,
    count: int = 0,
) -> dict:
    if labels is None:
        labels, count = build_object_mask(raw)
    metrics = {}
    metrics.update(a1_fold_amplitude(corrected, grid))
    metrics.update(a2_harmonic_power(corrected, grid))
    metrics.update(a3_uv_dependence(corrected, grid))
    metrics.update(a5_background_range(corrected, grid))
    metrics.update(a6_tile_level(corrected, grid))
    metrics.update(p1_spots(corrected, grid))
    metrics.update(preservation_metrics(raw, corrected, labels, count))
    return metrics


ARTIFACT_KEYS = (
    "A1_fold_amp_rel_pct",
    "A2_harmonic_mod_pct_max",
    "A3_uv_modulation_pct",
    "A5_bg_range_ratio",
    "A6_tile_level_sd_pct",
)
NOISE_FLOORS = {
    "A1_fold_amp_rel_pct": 0.30,
    "A2_harmonic_mod_pct_max": 0.61,
    "A3_uv_modulation_pct": 0.93,
    "A6_tile_level_sd_pct": 0.30,
}


def artifact_ratios(metrics: dict, baseline: dict) -> dict[str, float]:
    ratios = {}
    for key in ARTIFACT_KEYS:
        value, raw_value = metrics.get(key), baseline.get(key)
        if value is None or raw_value is None:
            continue
        if not np.isfinite(value) or not np.isfinite(raw_value):
            continue
        if key == "A5_bg_range_ratio":
            value, raw_value = value - 1.0, raw_value - 1.0
        if raw_value <= 1e-9:
            continue
        ratios[key] = float(
            max(value, NOISE_FLOORS.get(key, 1e-9)) / raw_value
        )
    return ratios


def artifact_index(metrics: dict, baseline: dict) -> float:
    ratios = list(artifact_ratios(metrics, baseline).values())
    if not ratios:
        return float("nan")
    return float(np.exp(np.mean(np.log(ratios))))


def physics_violations(model, raw: np.ndarray) -> list[str]:
    violations = []
    if isinstance(model, PeriodicFieldModel):
        flat = model.flat_tile
        if not np.isfinite(flat).all() or float(np.min(flat)) <= 0:
            violations.append("flatfield is non-finite or nonpositive")
        elif float(np.max(flat) / max(np.min(flat), 1e-9)) > 20:
            violations.append("flatfield dynamic range exceeds 20x")
        if np.ndim(model.dark) > 0 and np.any(np.asarray(model.dark)):
            dark_mean = float(np.mean(model.dark))
            image_floor = float(np.percentile(raw, 1))
            if dark_mean < 0:
                violations.append("darkfield mean is negative")
            if dark_mean >= image_floor:
                violations.append(
                    f"darkfield mean {dark_mean:.4g} reaches the image floor "
                    f"{image_floor:.4g}"
                )
    return violations


@dataclass
class CandidateResult:
    name: str
    model: object
    metrics: dict
    artifact_index: float
    violations: list[str]
    physics_violations: list[str]
    complexity: int
    fit_seconds: float = 0.0
    selection_score: float = float("inf")
    alternatives: list[dict] = field(default_factory=list)
    artifact_ratios: dict[str, float] = field(default_factory=dict)
    selection_samples: list[dict] = field(default_factory=list)
    score_log_se: float = 0.0
    pareto_optimal: bool = False
    fit_metrics: dict = field(default_factory=dict)
    validation_reports: list[dict] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.violations and not self.physics_violations

    def summary(self) -> dict:
        return {
            "name": self.name,
            "artifact_index": float(self.artifact_index),
            "selection_score": float(self.selection_score),
            "fit_seconds": float(self.fit_seconds),
            "valid": bool(self.valid),
            "pareto_optimal": bool(self.pareto_optimal),
            "score_log_se": float(self.score_log_se),
            "guardrail_violations": list(self.violations),
            "physics_violations": list(self.physics_violations),
            "artifact_ratios": dict(self.artifact_ratios),
            "metrics": dict(self.metrics),
            "fit_metrics": dict(self.fit_metrics),
            "validation_reports": list(self.validation_reports),
        }


def _selection_score(index: float, spot_uniformity, use_spot_uniformity: bool):
    if not np.isfinite(index) or index <= 0:
        return float("inf")
    spot_penalty = 0.0
    if (
        use_spot_uniformity
        and spot_uniformity is not None
        and np.isfinite(spot_uniformity)
    ):
        bounded = max(float(spot_uniformity), 1e-6)
        spot_penalty = 0.5 * abs(math.log(bounded))
    return float(index * math.exp(spot_penalty))


def _dominates(left: CandidateResult, right: CandidateResult) -> bool:
    keys = sorted(left.artifact_ratios)
    if not keys or set(keys) != set(right.artifact_ratios):
        return False
    no_worse = all(
        left.artifact_ratios[key] <= right.artifact_ratios[key] for key in keys
    )
    strictly_better = any(
        left.artifact_ratios[key] < right.artifact_ratios[key] for key in keys
    )
    return no_worse and strictly_better


def rank_candidates(
    candidates: Iterable[CandidateResult],
    tie_fraction: float = 0.05,
    use_spot_uniformity: bool = False,
) -> tuple[CandidateResult, list[CandidateResult]]:
    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        raise ValueError(
            "Every correction candidate failed a preservation or physics check"
        )
    for candidate in valid:
        candidate.selection_score = _selection_score(
            float(candidate.artifact_index),
            candidate.metrics.get("P1_spot_uniformity"),
            use_spot_uniformity,
        )
        sample_scores = [
            _selection_score(
                float(sample.get("artifact_index", float("nan"))),
                sample.get("P1_spot_uniformity"),
                use_spot_uniformity,
            )
            for sample in candidate.selection_samples
        ]
        sample_scores = [score for score in sample_scores if np.isfinite(score)]
        if len(sample_scores) >= 2:
            candidate.score_log_se = float(
                np.std(np.log(sample_scores), ddof=1) / math.sqrt(len(sample_scores))
            )

    require_complete_panel = any(candidate.name == "identity" for candidate in valid)
    scorable = [
        candidate
        for candidate in valid
        if np.isfinite(candidate.selection_score)
        and (
            not require_complete_panel
            or len(candidate.artifact_ratios) >= 3
            or not candidate.artifact_ratios
        )
    ]
    if not scorable:
        raise ValueError("No safe candidate produced a finite artifact score")

    frontier = [
        candidate
        for candidate in scorable
        if not any(
            other is not candidate and _dominates(other, candidate)
            for other in scorable
        )
    ]
    for candidate in frontier:
        candidate.pareto_optimal = True

    ranked = sorted(
        scorable,
        key=lambda c: (
            not c.pareto_optimal,
            c.selection_score,
            c.complexity,
            c.name,
        ),
    )
    selectable = sorted(
        frontier, key=lambda c: (c.selection_score, c.complexity, c.name)
    )
    best = selectable[0]
    best_score = best.selection_score
    base_margin = math.log1p(tie_fraction)
    tied = [
        candidate
        for candidate in selectable
        if math.log(candidate.selection_score / best_score)
        <= max(
            base_margin,
            1.96
            * math.hypot(best.score_log_se, candidate.score_log_se),
        )
    ]
    selected = min(tied, key=lambda c: (c.complexity, c.selection_score, c.name))
    return selected, ranked


def _candidate_specs(algorithm: str, darkfield_mode: str):
    if algorithm == ALGORITHM_OPTIONS[0]:
        return [
            ("identity", -1),
            ("basic_darkfield_off", 2),
            ("basic_darkfield_on", 3),
            ("fold_log_gradient", 1),
            ("split_half_affine", 0),
        ]
    if algorithm == ALGORITHM_OPTIONS[1]:
        if darkfield_mode == DARKFIELD_OPTIONS[1]:
            return [("basic_darkfield_on", 3)]
        if darkfield_mode == DARKFIELD_OPTIONS[2]:
            return [("basic_darkfield_off", 2)]
        return [("basic_darkfield_off", 2), ("basic_darkfield_on", 3)]
    if algorithm == ALGORITHM_OPTIONS[2]:
        return [("fold_log_gradient", 1)]
    if algorithm == ALGORITHM_OPTIONS[3]:
        return [("split_half_affine", 0)]
    raise ValueError(f"Unknown illumination-correction algorithm: {algorithm}")


def _aggregate_ratios(reports: list[dict]) -> dict[str, float]:
    output = {}
    for key in ARTIFACT_KEYS:
        values = [
            report["artifact_ratios"].get(key)
            for report in reports
            if report["artifact_ratios"].get(key) is not None
            and np.isfinite(report["artifact_ratios"].get(key))
            and report["artifact_ratios"].get(key) > 0
        ]
        if values:
            output[key] = float(np.exp(np.mean(np.log(values))))
    return output


def _aggregate_selection_metrics(reports: list[dict], basis: str) -> dict:
    output = {"selection_basis": basis, "selection_plane_count": len(reports)}
    keys = sorted(set().union(*(report["metrics"] for report in reports)))
    for key in keys:
        values = [report["metrics"].get(key) for report in reports]
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value)
        ]
        if numeric:
            output[key] = float(np.median(numeric))
    output["guardrail_violations"] = sorted(
        {
            violation
            for report in reports
            for violation in report["metrics"].get("guardrail_violations", [])
        }
    )
    output["guardrail_unavailable"] = sorted(
        {
            unavailable
            for report in reports
            for unavailable in report["metrics"].get("guardrail_unavailable", [])
        }
    )
    return output


def _evaluate_model_plane(
    model, raw: np.ndarray, grid: TileGrid, label: str
) -> dict:
    plane = as_plane(raw)
    if not np.isfinite(plane).all():
        raise ValueError(f"{label} contains non-finite source pixels")
    labels, count = build_object_mask(plane)
    baseline = evaluate(plane, plane, grid, labels, count)
    corrected = model.apply(plane)
    metrics = evaluate(corrected, plane, grid, labels, count)
    ratios = artifact_ratios(metrics, baseline)
    index = artifact_index(metrics, baseline)
    return {
        "label": label,
        "metrics": metrics,
        "artifact_ratios": ratios,
        "artifact_index": index,
    }


def select_model(
    raw: np.ndarray,
    grid: TileGrid,
    algorithm: str,
    darkfield_mode: str,
    per_tile_gain: bool,
    progress: Callable[[float, str, str], None] | None = None,
    validation_source: Callable[
        [], Iterable[tuple[str, np.ndarray]]
    ]
    | None = None,
    use_spot_uniformity: bool = False,
) -> CandidateResult:
    """Fit on one plane and select on independent planes when available."""
    raw = as_plane(raw)
    if not np.isfinite(raw).all():
        raise ValueError("The reference plane contains non-finite source pixels")
    labels, count = build_object_mask(raw)
    baseline = evaluate(raw, raw, grid, labels, count)
    validation_planes = (
        list(validation_source()) if validation_source is not None else []
    )

    if algorithm == ALGORITHM_OPTIONS[0] and not validation_planes:
        metrics = dict(baseline)
        metrics["selection_basis"] = "identity_without_holdout"
        model = IdentityModel(
            grid,
            diagnostics={
                "selection_basis": "identity_without_holdout",
                "reason": (
                    "Automatic correction requires an independent Z plane; "
                    "the channel was left unchanged"
                ),
            },
        )
        identity = CandidateResult(
            name="identity",
            model=model,
            metrics=metrics,
            artifact_index=1.0,
            violations=list(metrics["guardrail_violations"]),
            physics_violations=[],
            complexity=-1,
            artifact_ratios={key: 1.0 for key in ARTIFACT_KEYS},
            fit_metrics=dict(metrics),
            pareto_optimal=True,
            selection_score=1.0,
        )
        identity.alternatives = [identity.summary()]
        return identity

    specs = _candidate_specs(algorithm, darkfield_mode)
    candidates: list[CandidateResult] = []
    failures = []

    for index, (name, complexity) in enumerate(specs):
        if progress:
            progress(
                index / max(len(specs), 1),
                "Illumination correction",
                f"Fitting {name.replace('_', ' ')}",
        )
        started = time.monotonic()
        try:
            if name == "identity":
                model = IdentityModel(grid)
            elif name == "basic_darkfield_off":
                model = fit_basic(
                    raw, grid, darkfield=False, per_tile_gain=per_tile_gain
                )
            elif name == "basic_darkfield_on":
                model = fit_basic(
                    raw, grid, darkfield=True, per_tile_gain=per_tile_gain
                )
            elif name == "fold_log_gradient":
                model = fit_log_gradient(raw, grid, per_tile_gain=per_tile_gain)
            else:
                model = fit_split_half_affine(raw, grid)
            fit_report = _evaluate_model_plane(model, raw, grid, "fit plane")
            validation_reports = (
                [
                    _evaluate_model_plane(model, plane, grid, label)
                    for label, plane in validation_planes
                ]
                if validation_planes
                else []
            )
            selection_reports = validation_reports or [fit_report]
            ratios = _aggregate_ratios(selection_reports)
            index_value = (
                float(np.exp(np.mean(np.log(list(ratios.values())))))
                if ratios
                else float("nan")
            )
            basis = "held_out_z" if validation_reports else "fit_plane"
            metrics = _aggregate_selection_metrics(selection_reports, basis)
            violations = [
                f"{fit_report['label']}: {violation}"
                for violation in fit_report["metrics"]["guardrail_violations"]
            ]
            violations.extend(
                f"{report['label']}: {violation}"
                for report in validation_reports
                for violation in report["metrics"]["guardrail_violations"]
            )
            candidates.append(
                CandidateResult(
                    name=name,
                    model=model,
                    metrics=metrics,
                    artifact_index=index_value,
                    violations=violations,
                    physics_violations=physics_violations(model, raw),
                    complexity=complexity,
                    fit_seconds=time.monotonic() - started,
                    artifact_ratios=ratios,
                    selection_samples=[
                        {
                            "artifact_index": report["artifact_index"],
                            "P1_spot_uniformity": report["metrics"].get(
                                "P1_spot_uniformity"
                            ),
                        }
                        for report in selection_reports
                    ],
                    fit_metrics=fit_report["metrics"],
                    validation_reports=validation_reports,
                )
            )
        except Exception as exc:
            failures.append(f"{name}: {exc}")

    if not candidates:
        raise ValueError(
            "No correction candidate could be fitted. " + "; ".join(failures)
        )
    try:
        selected, _ = rank_candidates(
            candidates, use_spot_uniformity=use_spot_uniformity
        )
    except ValueError as exc:
        detail = "; ".join(
            f"{candidate.name}: "
            + ", ".join(candidate.violations + candidate.physics_violations)
            for candidate in candidates
        )
        raise ValueError(f"{exc}. {detail}") from exc

    selected.alternatives = [candidate.summary() for candidate in candidates]
    if failures:
        selected.alternatives.extend(
            {"name": failure.split(":", 1)[0], "valid": False, "error": failure}
            for failure in failures
        )
    return selected
