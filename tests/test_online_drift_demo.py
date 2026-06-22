"""Tests for the live-streaming online drift demo."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from drift_demo_common import demo_settings, reset_demo_tree  # noqa: E402
from run_online_drift_demo import (  # noqa: E402
    _MAX_DRIFT,
    drift_spec_for_tick,
    restamp_events,
    run_one_tick,
    run_online_drift_demo,
)

from nba.config import Settings  # noqa: E402
from nba.data.simulator import generate_logs  # noqa: E402
from nba.monitoring.retrain import RetrainLoop, bootstrap_deployed  # noqa: E402
from nba.ope.gate import PromotionGate  # noqa: E402


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        monitoring_report_path=tmp_path / "monitoring" / "drift_reports.jsonl",
        retrain_audit_path=tmp_path / "monitoring" / "retrain_audit.jsonl",
        deployed_model_manifest=tmp_path / "models" / "deployed.json",
        use_drift_monitoring=True,
        monitor_interval_events=1,
        monitor_recent_window=200,
        retrain_min_new_events=50,
    )


def test_drift_spec_ramp_monotonic() -> None:
    specs = [
        drift_spec_for_tick(t, mode="ramp", onset=2, ticks=6)
        for t in range(6)
    ]
    assert specs[0] is None
    assert specs[1] is None
    assert specs[2] is not None
    assert specs[5] is not None
    assert specs[2].reward_scale < specs[5].reward_scale
    assert specs[5].reward_scale == pytest.approx(_MAX_DRIFT.reward_scale)


def test_drift_spec_step_zero_before_onset() -> None:
    assert drift_spec_for_tick(1, mode="step", onset=3, ticks=8) is None
    step = drift_spec_for_tick(3, mode="step", onset=3, ticks=8)
    assert step is not None
    assert step.reward_scale == pytest.approx(_MAX_DRIFT.reward_scale)


def test_restamp_events_advances_timestamps() -> None:
    events = generate_logs(5, settings=Settings(), seed=1)
    clock = datetime(2026, 6, 22, 14, 0, tzinfo=UTC)
    stamped = restamp_events(events, clock)
    assert len(stamped) == 5
    assert stamped[0].timestamp == clock
    assert stamped[-1].timestamp == clock + timedelta(seconds=4)
    assert stamped[0].decision_id != events[0].decision_id
    assert stamped[0].reward == events[0].reward


def test_online_demo_appends_drift_report_per_tick(tmp_path: Path) -> None:
    settings = demo_settings(_settings(tmp_path))
    settings.ensure_dirs()
    reset_demo_tree(settings)

    records = run_online_drift_demo(
        warmup=400,
        events_per_tick=80,
        ticks=4,
        tick_seconds=0,
        drift_mode="step",
        drift_onset=1,
        seed=7,
        reset=False,
        send_email=False,
        settings=settings,
    )

    assert len(records) == 4
    drift_path = settings.monitoring_report_path
    lines = drift_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 4
    for line in lines:
        payload = json.loads(line)
        assert "signals" in payload


def test_run_one_tick_grows_event_store(tmp_path: Path) -> None:
    settings = demo_settings(_settings(tmp_path))
    settings.ensure_dirs()
    reset_demo_tree(settings)

    warmup = generate_logs(300, settings=settings, seed=1)
    from drift_demo_common import ingest_events_incremental  # noqa: E402

    ingest_events_incremental(settings, warmup)
    model, policy, _manifest, baseline_dr = bootstrap_deployed(
        settings=settings, events=warmup
    )
    loop = RetrainLoop(
        settings=settings,
        gate=PromotionGate(z=settings.ope_z, min_lift=settings.ope_min_lift),
    )

    outcome, cumulative, *_ = run_one_tick(
        tick=2,
        settings=settings,
        cumulative=list(warmup),
        deployed_model=model,
        deployed_policy=policy,
        baseline_dr=baseline_dr,
        loop=loop,
        promoted=False,
        events_per_tick=50,
        drift_mode="step",
        drift_onset=1,
        total_ticks=4,
        seed=3,
        send_email=False,
        clock=datetime(2026, 6, 22, 15, 0, tzinfo=UTC),
    )

    assert outcome.n_new_events == 50
    assert outcome.n_total_labeled == 350
    assert len(cumulative) == 350


def test_post_promote_tick_outcome_is_hold_not_promote(tmp_path: Path) -> None:
    """After promotion, monitor-only ticks must not report promoted=True."""
    settings = demo_settings(_settings(tmp_path))
    settings.ensure_dirs()
    reset_demo_tree(settings)

    warmup = generate_logs(300, settings=settings, seed=1)
    from drift_demo_common import ingest_events_incremental  # noqa: E402

    ingest_events_incremental(settings, warmup)
    model, policy, _manifest, baseline_dr = bootstrap_deployed(
        settings=settings, events=warmup
    )
    loop = RetrainLoop(
        settings=settings,
        gate=PromotionGate(z=settings.ope_z, min_lift=settings.ope_min_lift),
    )

    outcome, *_ = run_one_tick(
        tick=5,
        settings=settings,
        cumulative=list(warmup),
        deployed_model=model,
        deployed_policy=policy,
        baseline_dr=baseline_dr,
        loop=loop,
        promoted=True,
        events_per_tick=50,
        drift_mode="step",
        drift_onset=1,
        total_ticks=8,
        seed=3,
        send_email=False,
        clock=datetime(2026, 6, 22, 16, 0, tzinfo=UTC),
    )

    assert outcome.phase == "post_retrain"
    assert not outcome.promoted
