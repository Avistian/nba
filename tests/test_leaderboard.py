"""Tests for the append-only leaderboard: persistence, verdict logic, sign-normalized deltas,
and stable ranking."""

from __future__ import annotations

from pathlib import Path

from nba.config import Settings
from nba.eval.leaderboard import (
    ExperimentRecord,
    baseline_record,
    load_leaderboard,
    record_experiment,
    render_table,
)
from nba.eval.metrics import ExperimentMetrics


def _metrics(
    primary: float,
    *,
    lcb: float = 0.0,
    ope: float = 0.0,
    regret: float = 0.0,
    std: float = 0.0,
    route: float = 0.0,
    cvar: float = 0.0,
) -> ExperimentMetrics:
    return ExperimentMetrics(
        realized_shift_value_mean=primary,
        realized_shift_value_std=std,
        realized_shift_value_cvar=cvar,
        decision_regret_mean=regret,
        ope_value=ope,
        ope_lcb=lcb,
        optimality_gap=None,
        route_time_s_mean=route,
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        leaderboard_path=tmp_path / "leaderboard.jsonl",
        baseline_experiment_id="baseline",
        ope_min_lift=0.0,
    )


def _baseline(settings: Settings) -> ExperimentRecord:
    return record_experiment(
        _metrics(1.0, lcb=0.1, ope=0.1, regret=0.5),
        settings=settings,
        experiment_id="baseline",
        phase="baseline",
        flags={},
        baseline=None,
    )


def test_append_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _baseline(settings)
    record_experiment(
        _metrics(1.2, lcb=0.2, ope=0.2),
        settings=settings,
        experiment_id="exp-a",
        phase="10",
        flags={"NBA_FOO": "1"},
        baseline=baseline_record(load_leaderboard(settings.leaderboard_path), "baseline"),
    )

    lines = settings.leaderboard_path.read_text().splitlines()
    assert len(lines) == 2  # two appended rows

    records = load_leaderboard(settings.leaderboard_path)
    assert records[0].experiment_id == "baseline"
    assert records[0].metrics.realized_shift_value_mean == 1.0  # first row never mutated


def test_verdict_lift_requires_gate(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    base = _baseline(settings)
    # Beats primary AND clears the gate (lcb 0.3 > baseline ope 0.1).
    lift = record_experiment(
        _metrics(1.5, lcb=0.3, ope=0.3),
        settings=settings,
        experiment_id="exp-lift",
        phase="11",
        flags={},
        baseline=base,
    )
    assert lift.gate_passed
    assert lift.verdict == "lift"


def test_verdict_neutral_when_gate_fails(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    base = _baseline(settings)
    # Beats the mean but the lower bound does NOT clear the baseline => not real => neutral.
    neutral = record_experiment(
        _metrics(1.5, lcb=0.05, ope=0.3),
        settings=settings,
        experiment_id="exp-neutral",
        phase="11",
        flags={},
        baseline=base,
    )
    assert not neutral.gate_passed
    assert neutral.verdict == "neutral"


def test_verdict_regression_on_primary_drop(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    base = _baseline(settings)
    regression = record_experiment(
        _metrics(0.5, lcb=0.3, ope=0.3),  # primary drops below baseline's 1.0
        settings=settings,
        experiment_id="exp-reg",
        phase="11",
        flags={},
        baseline=base,
    )
    assert regression.verdict == "regression"


def test_deltas_sign_normalized(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    base = _baseline(settings)  # regret 0.5
    rec = record_experiment(
        _metrics(1.2, lcb=0.2, ope=0.2, regret=0.2),  # regret dropped 0.5 -> 0.2 (better)
        settings=settings,
        experiment_id="exp-d",
        phase="11",
        flags={},
        baseline=base,
    )
    # Lower-is-better metric: a drop is an improvement, so the normalized delta is positive.
    assert rec.deltas["decision_regret_mean"] > 0
    # Higher-is-better metric: a rise is an improvement, positive delta.
    assert rec.deltas["realized_shift_value_mean"] > 0


def test_render_table_ranks_best_first(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    base = _baseline(settings)
    record_experiment(
        _metrics(2.0, lcb=0.3, ope=0.3),
        settings=settings,
        experiment_id="top",
        phase="11",
        flags={},
        baseline=base,
    )
    record_experiment(
        _metrics(0.2, lcb=0.0, ope=0.0),
        settings=settings,
        experiment_id="bottom",
        phase="12",
        flags={},
        baseline=base,
    )
    table = render_table(load_leaderboard(settings.leaderboard_path))
    assert table.index("top") < table.index("bottom")  # best-first
