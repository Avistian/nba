"""Tests for monitor cadence gating (``monitor_interval_events``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from nba.api.store import EventStore
from nba.config import Settings
from nba.data.simulator import generate_logs
from nba.monitoring.cadence import count_new_labeled_since_last_monitor, evaluate_monitor_cadence
from nba.monitoring.exporter import build_snapshot, render_prometheus_text
from nba.monitoring.signals import DriftReport, DriftSignal, append_report


def _settings(tmp_path: Path, *, interval: int = 500) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        monitoring_report_path=tmp_path / "monitoring" / "drift_reports.jsonl",
        retrain_audit_path=tmp_path / "monitoring" / "retrain_audit.jsonl",
        deployed_model_manifest=tmp_path / "models" / "deployed.json",
        monitor_interval_events=interval,
    )


def _drift_report(*, at: datetime, n_labeled_total: int | None = None) -> DriftReport:
    signals = (
        DriftSignal("reward_psi", 0.05, 0.15, False, "ok"),
        DriftSignal("calibration_drift", 0.0, 0.05, False, "ok"),
        DriftSignal("feature_psi_max", 0.05, 0.20, False, "ok"),
        DriftSignal("overlap_health", 0.5, 0.02, False, "min_p=0.5"),
        DriftSignal("rolling_dr_drop", 0.0, 0.03, False, "ok"),
    )
    return DriftReport(
        timestamp=at,
        n_reference=100,
        n_recent=50,
        signals=signals,
        overlap_ok=True,
        n_labeled_total=n_labeled_total,
    )


def test_cadence_not_due_without_enough_labeled_events(tmp_path: Path) -> None:
    settings = _settings(tmp_path, interval=500)
    events = generate_logs(200, settings=settings, seed=1)
    cadence = evaluate_monitor_cadence(events, settings=settings)
    assert cadence.due is False
    assert cadence.events_since_last_report == 200
    assert cadence.interval == 500
    assert cadence.last_report_at is None


def test_cadence_due_on_first_run_after_interval(tmp_path: Path) -> None:
    settings = _settings(tmp_path, interval=500)
    events = generate_logs(600, settings=settings, seed=2)
    cadence = evaluate_monitor_cadence(events, settings=settings)
    assert cadence.due is True
    assert cadence.events_since_last_report == 600


def _stamp_monotonic(events: list) -> list:
    base = datetime(2026, 6, 1, tzinfo=UTC)
    return [
        e.model_copy(update={"timestamp": base + timedelta(minutes=i)})
        for i, e in enumerate(events)
    ]


def test_cadence_counts_only_events_after_last_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path, interval=10)
    events = _stamp_monotonic(generate_logs(40, settings=settings, seed=3))
    split_at = events[20].timestamp
    settings.monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)
    append_report(_drift_report(at=split_at), settings.monitoring_report_path)

    cadence = evaluate_monitor_cadence(events, settings=settings)
    assert cadence.last_report_at == split_at
    assert cadence.events_since_last_report == 19
    assert cadence.due is True


def test_cadence_not_due_when_few_events_since_last_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path, interval=10)
    events = _stamp_monotonic(generate_logs(40, settings=settings, seed=4))
    split_at = events[35].timestamp
    settings.monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)
    append_report(_drift_report(at=split_at), settings.monitoring_report_path)

    cadence = evaluate_monitor_cadence(events, settings=settings)
    assert cadence.events_since_last_report == 4
    assert cadence.due is False


def test_cadence_uses_n_labeled_total_when_present(tmp_path: Path) -> None:
    """When the last report stores ``n_labeled_total``, cadence uses log growth not timestamps."""
    settings = _settings(tmp_path, interval=500)
    events = generate_logs(1200, settings=settings, seed=8)
    settings.monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)
    append_report(
        _drift_report(at=datetime.now(UTC), n_labeled_total=700),
        settings.monitoring_report_path,
    )

    assert count_new_labeled_since_last_monitor(events, settings=settings) == 500

    cadence = evaluate_monitor_cadence(events, settings=settings)
    assert cadence.events_since_last_report == 500
    assert cadence.due is True


def test_exporter_emits_monitor_cadence_metrics(tmp_path: Path) -> None:
    settings = _settings(tmp_path, interval=25)
    events = generate_logs(30, settings=settings, seed=5)
    store = EventStore(settings.db_path)
    try:
        store.ingest_bandit_events(events, policy_name="test")
        snapshot = build_snapshot(settings=settings, store=store)
    finally:
        store.close()

    text = render_prometheus_text(snapshot, settings=settings)
    assert "nba_monitor_interval_events 25.0" in text
    assert "nba_monitor_events_since_last_report 30.0" in text
    assert "nba_monitor_due 1.0" in text


def test_exporter_cadence_due_zero_when_interval_not_met(tmp_path: Path) -> None:
    settings = _settings(tmp_path, interval=100)
    events = generate_logs(30, settings=settings, seed=6)
    report_time = datetime(2026, 5, 1, tzinfo=UTC)
    settings.monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)
    append_report(_drift_report(at=report_time), settings.monitoring_report_path)

    store = EventStore(settings.db_path)
    try:
        store.ingest_bandit_events(events, policy_name="test")
        snapshot = build_snapshot(settings=settings, store=store)
    finally:
        store.close()

    text = render_prometheus_text(snapshot, settings=settings)
    assert "nba_monitor_due 0.0" in text
    assert "nba_monitor_events_since_last_report 30.0" in text
