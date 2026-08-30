import sys
from pathlib import Path

import numpy as np


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from illumination import (  # noqa: E402
    CandidateResult,
    TileGrid,
    a1_fold_amplitude,
    choose_reference_grid,
    fit_grid,
    fit_log_gradient,
    fit_split_half_affine,
    rank_candidates,
)


def _synthetic_mosaic(seed=3, pitch=32, tiles=6):
    rng = np.random.default_rng(seed)
    size = pitch * tiles
    y, x = np.indices((size, size), dtype=np.float32)
    uy = (y % pitch) / pitch
    ux = (x % pitch) / pitch

    # A smooth, periodic illumination field with a pronounced seam falloff.
    flat = (
        0.72
        + 0.20 * np.sin(np.pi * uy) ** 2
        + 0.16 * np.sin(np.pi * ux) ** 2
        + 0.06 * np.sin(2 * np.pi * uy) * np.sin(2 * np.pi * ux)
    )

    # Independent texture in every physical tile prevents the estimator from
    # learning one repeated biological pattern.
    biology = 900.0 + rng.normal(0, 35, (size, size)).astype(np.float32)
    for tile_y in range(tiles):
        for tile_x in range(tiles):
            y0, x0 = tile_y * pitch, tile_x * pitch
            for _ in range(3):
                cy = y0 + int(rng.integers(4, pitch - 4))
                cx = x0 + int(rng.integers(4, pitch - 4))
                biology[cy - 2 : cy + 3, cx - 2 : cx + 3] += rng.uniform(250, 700)

    raw = 180.0 + flat * biology
    grid = TileGrid(
        pitch_y=float(pitch),
        pitch_x=float(pitch),
        seam_y=0.0,
        seam_x=0.0,
        height=size,
        width=size,
        seams_y=tuple(float(v) for v in range(0, size + 1, pitch)),
        seams_x=tuple(float(v) for v in range(0, size + 1, pitch)),
        seam_residual_y=0.0,
        seam_residual_x=0.0,
    )
    return biology.astype(np.float32), raw.astype(np.float32), grid


def test_fit_grid_recovers_periodic_geometry():
    _, raw, _ = _synthetic_mosaic()
    fitted = fit_grid(raw, pitch_min=24, pitch_max=40)

    assert abs(fitted.pitch_y - 32) < 2.0
    assert abs(fitted.pitch_x - 32) < 2.0
    assert fitted.is_valid
    assert len(fitted.seams_y) >= 4
    assert len(fitted.seams_x) >= 4


def test_auto_reference_uses_best_grid_in_dominant_channel_cluster(monkeypatch):
    class TileClient:
        tiles = {"frames": [{"IndexC": 0}, {"IndexC": 1}, {"IndexC": 2}]}

        @staticmethod
        def coordinatesToFrameIndex(xy, z, time, channel):
            return channel

        @staticmethod
        def getRegion(dataset_id, frame):
            return np.full((64, 64), frame, dtype=np.float32)

    def candidate_grid(image, pitch_min, pitch_max):
        channel = int(image[0, 0])
        pitches = ((32.0, 32.0), (32.5, 31.8), (20.0, 20.0))
        residuals = (1.0, 0.2, 0.01)
        prominences = (3.0, 8.0, 50.0)
        py, px = pitches[channel]
        return TileGrid(
            pitch_y=py,
            pitch_x=px,
            seam_y=0.0,
            seam_x=0.0,
            height=64,
            width=64,
            seams_y=(0.0, py, 2 * py, 3 * py),
            seams_x=(0.0, px, 2 * px, 3 * px),
            seam_residual_y=residuals[channel],
            seam_residual_x=residuals[channel],
            prominence_y=prominences[channel],
            prominence_x=prominences[channel],
        )

    monkeypatch.setattr("illumination.fit_grid", candidate_grid)
    selected, channel, reports = choose_reference_grid(
        TileClient(),
        "dataset-id",
        {"XY": 0, "Z": 0, "Time": 0},
        "Automatically choose best channel",
        0,
        10,
        40,
    )

    assert channel == 1
    assert selected.pitch_y == 32.5
    assert [report["cross_channel_agreement"] for report in reports] == [2, 2, 1]


def test_folded_log_gradient_reduces_position_locked_artifact():
    _, raw, grid = _synthetic_mosaic(seed=7)
    before = a1_fold_amplitude(raw, grid)["A1_fold_amp_rel_pct"]

    model = fit_log_gradient(raw, grid, n=64, per_tile_gain=True)
    corrected = model.apply(raw)
    after = a1_fold_amplitude(corrected, grid)["A1_fold_amp_rel_pct"]

    assert corrected.dtype == np.float32
    assert np.isfinite(corrected).all()
    assert after < before * 0.65


def test_split_half_affine_reduces_position_locked_artifact():
    _, raw, grid = _synthetic_mosaic(seed=11)
    before = a1_fold_amplitude(raw, grid)["A1_fold_amp_rel_pct"]

    model = fit_split_half_affine(
        raw,
        grid,
        profile_size=64,
        split_count=6,
        seed=11,
    )
    corrected = model.apply(raw)
    after = a1_fold_amplitude(corrected, grid)["A1_fold_amp_rel_pct"]

    assert np.isfinite(corrected).all()
    assert after < before * 0.75


def _candidate(name, artifact_index, p1=1.0, violations=(), complexity=0):
    return CandidateResult(
        name=name,
        model=object(),
        metrics={"P1_spot_uniformity": p1},
        artifact_index=artifact_index,
        violations=list(violations),
        physics_violations=[],
        complexity=complexity,
    )


def test_rank_candidates_rejects_guardrail_failures_and_penalizes_spot_bias():
    unsafe = _candidate(
        "basic_darkfield_on",
        artifact_index=0.18,
        p1=1.0,
        violations=("P2_spearman=0.96 < 0.98",),
    )
    biased = _candidate(
        "basic_darkfield_off", artifact_index=0.39, p1=1.55, complexity=2
    )
    stable = _candidate("fold_log_gradient", artifact_index=0.41, p1=1.02, complexity=1)

    selected, ranked = rank_candidates([unsafe, biased, stable])

    assert selected.name == "fold_log_gradient"
    assert unsafe not in ranked


def test_rank_candidates_prefers_simpler_method_inside_tie_margin():
    complex_candidate = _candidate("basic_darkfield_on", 0.200, complexity=3)
    simple_candidate = _candidate("split_half_affine", 0.207, complexity=0)

    selected, _ = rank_candidates(
        [complex_candidate, simple_candidate], tie_fraction=0.05
    )

    assert selected.name == "split_half_affine"
