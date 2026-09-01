"""Seeded, translation-only position refinement for raw acquisition tiles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


DEFAULT_NCC_THRESHOLD = 0.5
DEFAULT_COARSE_RADIUS = 24
DEFAULT_COARSE_STEP = 3
DEFAULT_FINE_RADIUS = 4
DEFAULT_FINE_STEP = 1
DEFAULT_STAGE_BIN_GAP_UM = 50.0
MIN_OVERLAP_DIMENSION = 30


@dataclass(frozen=True)
class TileEdge:
    first: int
    second: int
    axis: str


@dataclass(frozen=True)
class PairMeasurement:
    first: int
    second: int
    axis: str
    predicted_shift_x: int
    predicted_shift_y: int
    shift_x: int
    shift_y: int
    ncc: float
    accepted: bool

    def as_dict(self) -> dict[str, int | float | bool | str]:
        return {
            "first": self.first,
            "second": self.second,
            "axis": self.axis,
            "predicted_shift_x": self.predicted_shift_x,
            "predicted_shift_y": self.predicted_shift_y,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
            "ncc": self.ncc,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class RefinementResult:
    positions: np.ndarray
    measurements: tuple[PairMeasurement, ...]
    accepted_measurements: tuple[PairMeasurement, ...]
    residuals: tuple[float, ...]
    max_residual: float | None
    similarity_matrix: np.ndarray

    def as_dict(self) -> dict:
        shifts = self.positions.astype(np.float64)
        return {
            "pairs_total": len(self.measurements),
            "pairs_matched": len(self.accepted_measurements),
            "max_residual_px": self.max_residual,
            "pair_residuals_px": list(self.residuals),
            "pair_measurements": [item.as_dict() for item in self.measurements],
            "similarity_matrix": self.similarity_matrix.tolist(),
            "positions": shifts.astype(int).tolist(),
        }


def _axis_bins(values: np.ndarray, gap_um: float) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    bins = np.zeros(values.size, dtype=np.int64)
    current = 0
    for sorted_index, source_index in enumerate(order):
        if (
            sorted_index > 0
            and values[source_index] - values[order[sorted_index - 1]] > gap_um
        ):
            current += 1
        bins[source_index] = current
    return bins


def build_adjacency(
    stage_positions_um: np.ndarray,
    gap_um: float = DEFAULT_STAGE_BIN_GAP_UM,
) -> tuple[TileEdge, ...]:
    """Build a four-neighbor graph from jittered microscope stage coordinates."""
    stages = np.asarray(stage_positions_um, dtype=np.float64)
    if stages.ndim != 2 or stages.shape[1] != 2:
        raise ValueError(f"stage positions must have shape (tiles, 2), got {stages.shape}")
    if stages.shape[0] == 0 or not np.all(np.isfinite(stages)):
        raise ValueError("stage positions must be nonempty and finite")
    if not np.isfinite(gap_um) or gap_um <= 0:
        raise ValueError("stage-bin gap must be positive and finite")

    columns = _axis_bins(stages[:, 0], gap_um)
    rows = _axis_bins(stages[:, 1], gap_um)
    grid: dict[tuple[int, int], int] = {}
    for index, (row, column) in enumerate(zip(rows, columns, strict=True)):
        key = int(row), int(column)
        if key in grid:
            raise ValueError(
                "stage binning assigned multiple acquisition positions to grid cell "
                f"{key}: {grid[key]} and {index}"
            )
        grid[key] = index

    edges: list[TileEdge] = []
    for (row, column), first in sorted(grid.items()):
        right = grid.get((row, column + 1))
        if right is not None:
            edges.append(TileEdge(first, right, "horizontal"))
        below = grid.get((row + 1, column))
        if below is not None:
            edges.append(TileEdge(first, below, "vertical"))
    return tuple(edges)


def overlap_ncc(
    first: np.ndarray,
    second: np.ndarray,
    shift_x: int,
    shift_y: int,
    min_overlap_dimension: int = MIN_OVERLAP_DIMENSION,
) -> float | None:
    """Return NCC where ``second[y, x] ~= first[y + Sy, x + Sx]``."""
    first_values = np.asarray(first)
    second_values = np.asarray(second)
    if first_values.ndim != 2 or second_values.ndim != 2:
        raise ValueError("pairwise refinement requires two-dimensional tile images")
    if first_values.shape != second_values.shape:
        raise ValueError(
            f"tile images must share a shape, got {first_values.shape} and "
            f"{second_values.shape}"
        )
    height, width = first_values.shape
    first_x0 = max(0, int(shift_x))
    first_x1 = min(width, int(shift_x) + width)
    first_y0 = max(0, int(shift_y))
    first_y1 = min(height, int(shift_y) + height)
    if (
        first_x1 - first_x0 < min_overlap_dimension
        or first_y1 - first_y0 < min_overlap_dimension
    ):
        return None

    first_overlap = first_values[first_y0:first_y1, first_x0:first_x1]
    second_overlap = second_values[
        first_y0 - shift_y : first_y1 - shift_y,
        first_x0 - shift_x : first_x1 - shift_x,
    ]
    finite = np.isfinite(first_overlap) & np.isfinite(second_overlap)
    if np.count_nonzero(finite) < min_overlap_dimension**2:
        return None
    first_vector = first_overlap[finite].astype(np.float64, copy=False)
    second_vector = second_overlap[finite].astype(np.float64, copy=False)
    first_vector = first_vector - float(np.mean(first_vector))
    second_vector = second_vector - float(np.mean(second_vector))
    denominator = float(
        np.sqrt(np.dot(first_vector, first_vector) * np.dot(second_vector, second_vector))
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        return 0.0
    score = float(np.dot(first_vector, second_vector) / denominator)
    return float(np.clip(score, -1.0, 1.0))


def _search(
    first: np.ndarray,
    second: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    step: int,
) -> tuple[float, int, int]:
    best_score = -np.inf
    best_x = int(center_x)
    best_y = int(center_y)
    for shift_y in range(center_y - radius, center_y + radius + 1, step):
        for shift_x in range(center_x - radius, center_x + radius + 1, step):
            score = overlap_ncc(first, second, shift_x, shift_y)
            if score is not None and score > best_score:
                best_score = score
                best_x = shift_x
                best_y = shift_y
    if not np.isfinite(best_score):
        return 0.0, int(center_x), int(center_y)
    return float(best_score), best_x, best_y


def measure_pair(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_index: int,
    second_index: int,
    predicted_shift: tuple[int, int],
    axis: str = "unknown",
    ncc_threshold: float = DEFAULT_NCC_THRESHOLD,
    coarse_radius: int = DEFAULT_COARSE_RADIUS,
    coarse_step: int = DEFAULT_COARSE_STEP,
    fine_radius: int = DEFAULT_FINE_RADIUS,
    fine_step: int = DEFAULT_FINE_STEP,
) -> PairMeasurement:
    """Measure one content offset with the validated seeded coarse/fine search."""
    predicted_x, predicted_y = (int(value) for value in predicted_shift)
    _, coarse_x, coarse_y = _search(
        first,
        second,
        predicted_x,
        predicted_y,
        int(coarse_radius),
        int(coarse_step),
    )
    score, shift_x, shift_y = _search(
        first,
        second,
        coarse_x,
        coarse_y,
        int(fine_radius),
        int(fine_step),
    )
    return PairMeasurement(
        first=int(first_index),
        second=int(second_index),
        axis=str(axis),
        predicted_shift_x=predicted_x,
        predicted_shift_y=predicted_y,
        shift_x=shift_x,
        shift_y=shift_y,
        ncc=score,
        accepted=bool(score >= float(ncc_threshold)),
    )


def _target_delta(
    measurement: PairMeasurement, linear_transform: np.ndarray
) -> np.ndarray:
    shift = np.asarray((measurement.shift_x, measurement.shift_y), dtype=np.float64)
    return linear_transform @ shift


def _fit_similarity(
    original_positions: np.ndarray,
    measurements: Sequence[PairMeasurement],
    linear_transform: np.ndarray,
) -> np.ndarray:
    if not measurements:
        return np.eye(2, dtype=np.float64)
    rows = []
    responses = []
    for measurement in measurements:
        source = original_positions[measurement.second] - original_positions[measurement.first]
        target = _target_delta(measurement, linear_transform)
        weight = np.sqrt(max(measurement.ncc, np.finfo(np.float64).eps))
        dx, dy = source
        rows.extend((weight * np.asarray((dx, -dy)), weight * np.asarray((dy, dx))))
        responses.extend((weight * target[0], weight * target[1]))
    design = np.asarray(rows, dtype=np.float64)
    response = np.asarray(responses, dtype=np.float64)
    if np.linalg.matrix_rank(design) < 2:
        return np.eye(2, dtype=np.float64)
    (scale_cos, scale_sin), *_ = np.linalg.lstsq(design, response, rcond=None)
    matrix = np.asarray(
        ((scale_cos, -scale_sin), (scale_sin, scale_cos)), dtype=np.float64
    )
    if not np.all(np.isfinite(matrix)):
        return np.eye(2, dtype=np.float64)
    return matrix


def _components(
    count: int, measurements: Sequence[PairMeasurement]
) -> list[list[int]]:
    neighbors = [set() for _ in range(count)]
    for measurement in measurements:
        neighbors[measurement.first].add(measurement.second)
        neighbors[measurement.second].add(measurement.first)
    components = []
    unseen = set(range(count))
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(neighbors[node], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def solve_positions(
    original_positions: np.ndarray,
    measurements: Sequence[PairMeasurement],
    linear_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Globally solve confident translation constraints with similarity fallback."""
    original = np.asarray(original_positions, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 2 or original.shape[0] == 0:
        raise ValueError(f"positions must have shape (tiles, 2), got {original.shape}")
    if not np.all(np.isfinite(original)):
        raise ValueError("positions must be finite")
    transform = (
        -np.eye(2, dtype=np.float64)
        if linear_transform is None
        else np.asarray(linear_transform, dtype=np.float64)
    )
    if transform.shape != (2, 2) or not np.all(np.isfinite(transform)):
        raise ValueError("the shared source transform must be a finite 2x2 matrix")
    if abs(float(np.linalg.det(transform))) <= np.finfo(np.float64).eps:
        raise ValueError("the shared source transform must be invertible")

    accepted = tuple(item for item in measurements if item.accepted)
    similarity = _fit_similarity(original, accepted, transform)
    center = np.mean(original, axis=0)
    fallback = (original - center) @ similarity.T + center
    fallback -= np.mean(fallback - original, axis=0, keepdims=True)

    solved = np.empty_like(original)
    for component in _components(original.shape[0], accepted):
        component_set = set(component)
        component_measurements = [
            item
            for item in accepted
            if item.first in component_set and item.second in component_set
        ]
        if not component_measurements:
            solved[component] = fallback[component]
            continue
        local_index = {source: local for local, source in enumerate(component)}
        design = np.zeros((len(component_measurements) + 1, len(component)), dtype=np.float64)
        targets = np.zeros((len(component_measurements) + 1, 2), dtype=np.float64)
        for row, measurement in enumerate(component_measurements):
            weight = np.sqrt(max(measurement.ncc, np.finfo(np.float64).eps))
            design[row, local_index[measurement.first]] = -weight
            design[row, local_index[measurement.second]] = weight
            targets[row] = weight * _target_delta(measurement, transform)
        # Incidence rows sum to zero, so this fixes the component translation without
        # changing any edge fit. Anchor at the similarity fallback's component mean.
        design[-1] = 1.0
        targets[-1] = 0.0
        relative, *_ = np.linalg.lstsq(design, targets, rcond=None)
        relative -= np.mean(relative, axis=0, keepdims=True)
        solved[component] = relative + np.mean(fallback[component], axis=0)

    solved -= np.mean(solved - original, axis=0, keepdims=True)
    rounded = np.rint(solved).astype(np.int64)
    rounded -= np.rint(np.mean(rounded - original, axis=0)).astype(np.int64)
    return rounded, similarity


def refine_positions(
    tiles: Sequence[np.ndarray],
    original_positions: np.ndarray,
    stage_positions_um: np.ndarray,
    *,
    linear_transform: np.ndarray | None = None,
    ncc_threshold: float = DEFAULT_NCC_THRESHOLD,
    progress: Callable[[int, int, TileEdge], None] | None = None,
) -> RefinementResult:
    """Measure adjacent pairs and return globally consistent integer positions."""
    original = np.asarray(original_positions, dtype=np.float64)
    stages = np.asarray(stage_positions_um, dtype=np.float64)
    if len(tiles) != original.shape[0] or stages.shape != original.shape:
        raise ValueError(
            "tile images, original positions, and stage positions must have equal counts"
        )
    transform = (
        -np.eye(2, dtype=np.float64)
        if linear_transform is None
        else np.asarray(linear_transform, dtype=np.float64)
    )
    inverse_transform = np.linalg.inv(transform)
    edges = build_adjacency(stages)
    measurements = []
    for edge_index, edge in enumerate(edges):
        placement_delta = original[edge.second] - original[edge.first]
        predicted = np.rint(inverse_transform @ placement_delta).astype(np.int64)
        measurements.append(
            measure_pair(
                tiles[edge.first],
                tiles[edge.second],
                first_index=edge.first,
                second_index=edge.second,
                predicted_shift=(int(predicted[0]), int(predicted[1])),
                axis=edge.axis,
                ncc_threshold=ncc_threshold,
            )
        )
        if progress is not None:
            progress(edge_index + 1, len(edges), edge)

    accepted = tuple(item for item in measurements if item.accepted)
    positions, similarity = solve_positions(original, accepted, transform)
    residuals = tuple(
        float(
            np.linalg.norm(
                positions[item.second]
                - positions[item.first]
                - _target_delta(item, transform)
            )
        )
        for item in accepted
    )
    return RefinementResult(
        positions=positions,
        measurements=tuple(measurements),
        accepted_measurements=accepted,
        residuals=residuals,
        max_residual=max(residuals) if residuals else None,
        similarity_matrix=similarity,
    )
