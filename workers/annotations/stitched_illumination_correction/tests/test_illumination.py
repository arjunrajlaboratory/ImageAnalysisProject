import sys
from pathlib import Path

import numpy as np
import pytest


WORKER_DIR = Path(__file__).resolve().parents[1]
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from illumination import (  # noqa: E402
    CandidateResult,
    IdentityModel,
    TileGrid,
    a1_fold_amplitude,
    choose_reference_grid,
    fit_basic,
    fit_grid,
    fit_log_gradient,
    fit_split_half_affine,
    normalize_flat,
    p2_object_intensity,
    preservation_metrics,
    rank_candidates,
    select_model,
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


def _candidate(
    name,
    artifact_index,
    p1=1.0,
    violations=(),
    complexity=0,
    artifact_ratios=None,
    sample_scores=(),
):
    return CandidateResult(
        name=name,
        model=object(),
        metrics={"P1_spot_uniformity": p1},
        artifact_index=artifact_index,
        violations=list(violations),
        physics_violations=[],
        complexity=complexity,
        artifact_ratios=artifact_ratios or {},
        selection_samples=[
            {"artifact_index": score, "P1_spot_uniformity": 1.0}
            for score in sample_scores
        ],
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

    selected, ranked = rank_candidates(
        [unsafe, biased, stable], use_spot_uniformity=True
    )

    assert selected.name == "fold_log_gradient"
    assert unsafe not in ranked


def test_rank_candidates_prefers_simpler_method_inside_tie_margin():
    complex_candidate = _candidate("basic_darkfield_on", 0.200, complexity=3)
    simple_candidate = _candidate("split_half_affine", 0.207, complexity=0)

    selected, _ = rank_candidates(
        [complex_candidate, simple_candidate], tie_fraction=0.05
    )

    assert selected.name == "split_half_affine"


def test_automatic_selection_can_keep_identity_when_corrections_are_worse():
    identity = _candidate("identity", 1.0, complexity=-1)
    worse = _candidate("fold_log_gradient", 1.25, complexity=1)

    selected, _ = rank_candidates([identity, worse])

    assert selected.name == "identity"


def test_identity_is_kept_when_pareto_improvement_is_inside_tie_margin():
    identity = _candidate(
        "identity",
        1.0,
        complexity=-1,
        artifact_ratios={key: 1.0 for key in "abcde"},
        sample_scores=(1.0, 1.0, 1.0),
    )
    marginal = _candidate(
        "fold_log_gradient",
        0.99,
        complexity=1,
        artifact_ratios={key: 0.99 for key in "abcde"},
        sample_scores=(0.98, 1.0, 0.99),
    )

    selected, _ = rank_candidates([identity, marginal], tie_fraction=0.05)

    assert selected.name == "identity"


def test_identity_requires_consistent_held_out_improvement():
    identity = _candidate(
        "identity",
        1.0,
        complexity=-1,
        sample_scores=(1.0, 1.0, 1.0),
    )
    inconsistent = _candidate(
        "fold_log_gradient",
        0.80,
        complexity=1,
        sample_scores=(0.70, 1.05, 0.70),
    )

    selected, _ = rank_candidates([identity, inconsistent], tie_fraction=0.05)

    assert selected.name == "identity"


def test_identity_can_be_displaced_by_strong_consistent_improvement():
    identity = _candidate(
        "identity",
        1.0,
        complexity=-1,
        sample_scores=(1.0, 1.0, 1.0),
    )
    improved = _candidate(
        "fold_log_gradient",
        0.75,
        complexity=1,
        sample_scores=(0.70, 0.80, 0.75),
    )

    selected, _ = rank_candidates([identity, improved], tie_fraction=0.05)

    assert selected.name == "fold_log_gradient"


def test_identity_gate_filters_inconsistent_model_before_tie_breaking():
    identity = _candidate(
        "identity",
        1.0,
        complexity=-1,
        sample_scores=(1.0, 1.0, 1.0),
    )
    consistent = _candidate(
        "basic_darkfield_off",
        0.70,
        complexity=2,
        sample_scores=(0.70, 0.70, 0.70),
    )
    simpler_but_inconsistent = _candidate(
        "split_half_affine",
        0.72,
        complexity=0,
        sample_scores=(0.60, 1.05, 0.60),
    )

    selected, _ = rank_candidates(
        [identity, consistent, simpler_but_inconsistent], tie_fraction=0.05
    )

    assert selected.name == "basic_darkfield_off"


def test_identity_gate_filters_inconsistent_model_before_pareto_selection():
    identity = _candidate(
        "identity",
        1.0,
        complexity=-1,
        artifact_ratios={key: 1.0 for key in "abcde"},
        sample_scores=(1.0, 1.0, 1.0),
    )
    consistent = _candidate(
        "basic_darkfield_off",
        0.75,
        complexity=2,
        artifact_ratios={key: 0.80 for key in "abcde"},
        sample_scores=(0.70, 0.75, 0.80),
    )
    dominating_but_inconsistent = _candidate(
        "split_half_affine",
        0.70,
        complexity=0,
        artifact_ratios={key: 0.70 for key in "abcde"},
        sample_scores=(0.60, 1.05, 0.60),
    )

    selected, _ = rank_candidates(
        [identity, consistent, dominating_but_inconsistent], tie_fraction=0.05
    )

    assert selected.name == "basic_darkfield_off"


def test_candidate_uncertainty_cannot_expand_the_fixed_tie_margin():
    best = _candidate(
        "basic_darkfield_on",
        1.0,
        complexity=3,
        sample_scores=(1.0, 1.0, 1.0),
    )
    noisy_but_worse = _candidate(
        "split_half_affine",
        1.10,
        complexity=0,
        sample_scores=(0.5, 1.1, 2.2),
    )

    selected, _ = rank_candidates([best, noisy_but_worse], tie_fraction=0.05)

    assert selected.name == "basic_darkfield_on"


def test_rank_candidates_rejects_undefined_artifact_scores():
    undefined = _candidate("split_half_affine", float("nan"), complexity=0)

    with pytest.raises(ValueError, match="finite artifact score"):
        rank_candidates([undefined])


def test_spot_penalty_treats_zero_as_extreme_bias():
    zero_outer = _candidate("zero_outer", 0.20, p1=0.0, complexity=0)
    balanced = _candidate("balanced", 0.25, p1=1.0, complexity=1)

    selected, _ = rank_candidates(
        [zero_outer, balanced], use_spot_uniformity=True
    )

    assert selected.name == "balanced"


def test_normalize_flat_rejects_nonfinite_and_nonpositive_fields():
    with pytest.raises(ValueError, match="finite and strictly positive"):
        normalize_flat(np.full((8, 8), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="finite and strictly positive"):
        normalize_flat(np.zeros((8, 8), dtype=np.float32))


def test_range_guardrail_ignores_preexisting_zeros_but_rejects_new_ones():
    raw = np.full((64, 64), 10.0, dtype=np.float32)
    raw[:8] = 0.0
    unchanged = preservation_metrics(raw, raw.copy())

    damaged = raw.copy()
    damaged[8:16] = 0.0
    damaged_metrics = preservation_metrics(raw, damaged)

    assert unchanged["P5_frac_new_nonpositive"] == 0.0
    assert not any(
        "P5_frac_new_nonpositive" in item
        for item in unchanged["guardrail_violations"]
    )
    assert any(
        "P5_frac_new_nonpositive" in item
        for item in damaged_metrics["guardrail_violations"]
    )


def test_range_guardrail_rejects_nonfinite_output():
    raw = np.full((64, 64), 10.0, dtype=np.float32)
    corrected = raw.copy()
    corrected[0, 0] = np.nan

    metrics = preservation_metrics(raw, corrected)

    assert metrics["P5_frac_nonfinite"] > 0
    assert any(
        "P5_frac_nonfinite" in item for item in metrics["guardrail_violations"]
    )


def test_constant_object_intensities_make_rank_guardrail_inapplicable():
    labels = np.arange(1, 11, dtype=np.int32).reshape(2, 5)
    raw = np.ones(labels.shape, dtype=np.float32)
    corrected = np.ones(labels.shape, dtype=np.float32)

    metrics = p2_object_intensity(raw, corrected, labels, count=10)

    assert metrics["P2_applicable"] is False
    assert np.isnan(metrics["P2_spearman"])


def test_constant_corrected_intensities_are_a_rank_guardrail_violation():
    labels = np.arange(1, 11, dtype=np.int32).reshape(2, 5)
    raw = np.arange(1, 11, dtype=np.float32).reshape(2, 5)
    corrected = np.ones(labels.shape, dtype=np.float32)

    metrics = p2_object_intensity(raw, corrected, labels, count=10)

    assert metrics["P2_applicable"] is True
    assert metrics["P2_spearman"] == 0.0
    assert metrics["P2_corrected_constant"] is True


@pytest.mark.parametrize("erased_value", [0.0, -1.0])
def test_nonpositive_corrected_object_sum_is_a_preservation_violation(
    erased_value,
):
    labels = np.arange(1, 11, dtype=np.int32).reshape(2, 5)
    raw = np.arange(1, 11, dtype=np.float32).reshape(2, 5)
    corrected = raw.copy()
    corrected[0, 0] = erased_value

    metrics = preservation_metrics(raw, corrected, labels, count=10)

    assert metrics["P2_n_objects"] == 10
    assert metrics["P2_n_erased_objects"] == 1
    assert metrics["P2_frac_erased_objects"] == pytest.approx(0.1)
    assert any(
        "P2_frac_erased_objects" in item
        for item in metrics["guardrail_violations"]
    )


def test_erased_object_fails_when_rank_correlation_is_unavailable():
    labels = np.arange(1, 6, dtype=np.int32).reshape(1, 5)
    raw = np.arange(1, 6, dtype=np.float32).reshape(1, 5)
    corrected = raw.copy()
    corrected[0, 0] = 0.0

    metrics = preservation_metrics(raw, corrected, labels, count=5)

    assert metrics["P2_applicable"] is False
    assert metrics["P2_frac_erased_objects"] == pytest.approx(0.2)
    assert any(
        "P2_frac_erased_objects" in item
        for item in metrics["guardrail_violations"]
    )


def test_automatic_selection_without_held_out_plane_returns_identity():
    _, raw, grid = _synthetic_mosaic(seed=17)

    selected = select_model(
        raw,
        grid,
        "Automatic (recommended)",
        "Automatic",
        per_tile_gain=False,
    )

    assert isinstance(selected.model, IdentityModel)
    assert selected.name == "identity"
    assert selected.metrics["selection_basis"] == "identity_without_holdout"


def test_automatic_selection_fails_when_every_correction_algorithm_errors(
    monkeypatch,
):
    _, raw, grid = _synthetic_mosaic(seed=23)

    def unavailable(*args, **kwargs):
        raise ValueError("not applicable to this image")

    monkeypatch.setattr("illumination.fit_basic", unavailable)
    monkeypatch.setattr("illumination.fit_log_gradient", unavailable)
    monkeypatch.setattr("illumination.fit_split_half_affine", unavailable)

    with pytest.raises(ValueError, match="Every non-identity correction"):
        select_model(
            raw,
            grid,
            "Automatic (recommended)",
            "Automatic",
            per_tile_gain=False,
            validation_source=lambda: [("held-out Z 2", raw.copy())],
        )


def test_automatic_selection_fails_when_identity_baseline_errors(monkeypatch):
    _, raw, grid = _synthetic_mosaic(seed=31)

    monkeypatch.setattr("illumination.fit_basic", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "illumination.fit_log_gradient", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "illumination.fit_split_half_affine", lambda *args, **kwargs: object()
    )

    def evaluate_model(model, plane, fitted_grid, label):
        if isinstance(model, IdentityModel):
            raise RuntimeError("identity evaluation crashed")
        return {
            "label": label,
            "metrics": {"guardrail_violations": [], "guardrail_unavailable": []},
            "artifact_ratios": {key: 0.5 for key in "abcde"},
            "artifact_index": 0.5,
        }

    monkeypatch.setattr("illumination._evaluate_model_plane", evaluate_model)

    with pytest.raises(ValueError, match="identity baseline failed"):
        select_model(
            raw,
            grid,
            "Automatic (recommended)",
            "Automatic",
            per_tile_gain=False,
            validation_source=lambda: [("held-out Z 2", raw.copy())],
        )


def test_partial_automatic_failures_distinguish_unavailable_from_errors(
    monkeypatch,
):
    _, raw, grid = _synthetic_mosaic(seed=29)

    def crashed(*args, **kwargs):
        raise RuntimeError("dependency crashed")

    def unavailable(*args, **kwargs):
        raise ValueError("not applicable to this image")

    monkeypatch.setattr("illumination.fit_basic", crashed)
    monkeypatch.setattr(
        "illumination.fit_log_gradient",
        lambda *args, **kwargs: IdentityModel(grid),
    )
    monkeypatch.setattr("illumination.fit_split_half_affine", unavailable)

    selected = select_model(
        raw,
        grid,
        "Automatic (recommended)",
        "Automatic",
        per_tile_gain=False,
        validation_source=lambda: [("held-out Z 2", raw.copy())],
    )

    assert {failure["kind"] for failure in selected.candidate_failures} == {
        "error",
        "unavailable",
    }
    assert all("exception_type" in failure for failure in selected.candidate_failures)


@pytest.mark.parametrize("darkfield", [False, True])
def test_basic_fit_smoke_uses_production_dependency_path(darkfield):
    _, raw, grid = _synthetic_mosaic(seed=19, pitch=32, tiles=4)

    model = fit_basic(
        raw,
        grid,
        darkfield=darkfield,
        per_tile_gain=False,
        tile_n=32,
    )
    corrected = model.apply(raw)

    assert corrected.shape == raw.shape
    assert np.isfinite(corrected).all()
