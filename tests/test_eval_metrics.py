"""Tests for the leaderboard metric set: determinism and sane bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nba.config import Settings
from nba.eval.metrics import ExperimentMetrics, evaluate
from nba.schema import REWARD

_REWARD_MAX = max(REWARD.values())


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(data_dir=tmp_path / "data", n_bootstrap=2, **overrides)


def test_evaluate_is_deterministic_and_bounded(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    a = evaluate(settings=settings, n_shifts=1, seeds=(7,), n_logs=900, shift=12, replan_every=6)
    b = evaluate(settings=settings, n_shifts=1, seeds=(7,), n_logs=900, shift=12, replan_every=6)
    assert a == b

    assert isinstance(a, ExperimentMetrics)
    assert a.realized_shift_value_std >= 0.0
    assert a.decision_regret_mean >= 0.0
    # Realized value sums per-door true rewards over the shift; bounded by shift * max reward.
    assert a.realized_shift_value_mean <= 12 * _REWARD_MAX + 1e-9
    assert a.route_time_s_mean >= 0.0


def test_relational_metrics_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path, dataset_mode="relational")
    metrics = evaluate(settings=settings, n_shifts=1, seeds=(7,), n_logs=900, shift=12)
    assert metrics.decision_regret_mean >= 0.0
