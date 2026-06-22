"""Tests for significant-drift email alerting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from nba.config import Settings
from nba.monitoring.alerting import (
    AlertDecision,
    EmailAlerter,
    format_drift_alert,
    is_significant,
    maybe_alert_drift,
)
from nba.monitoring.retrain import RetrainOutcome
from nba.monitoring.signals import DriftReport, DriftSignal
from nba.monitoring.triggers import RetrainTrigger


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "data_dir": tmp_path / "data",
        "model_dir": tmp_path / "models",
        "db_path": tmp_path / "events.db",
        "monitoring_report_path": tmp_path / "monitoring" / "drift_reports.jsonl",
        "retrain_audit_path": tmp_path / "monitoring" / "retrain_audit.jsonl",
        "deployed_model_manifest": tmp_path / "models" / "deployed.json",
    }
    base.update(overrides)
    return Settings(**base)


def _report(*, reward_triggered: bool = True) -> DriftReport:
    signals = (
        DriftSignal(
            "reward_psi",
            0.25 if reward_triggered else 0.05,
            0.15,
            reward_triggered,
            "PSI",
        ),
        DriftSignal("calibration_drift", 0.0, 0.05, False, "ok"),
        DriftSignal("feature_psi_max", 0.05, 0.20, False, "ok"),
        DriftSignal("overlap_health", 0.5, 0.02, False, "min_p=0.5"),
        DriftSignal("rolling_dr_drop", 0.0, 0.03, False, "ok"),
    )
    return DriftReport(
        timestamp=datetime.now(UTC),
        n_reference=100,
        n_recent=50,
        signals=signals,
        overlap_ok=True,
    )


def _trigger(*, should: bool = True) -> RetrainTrigger:
    return RetrainTrigger(
        should_retrain=should,
        reasons=("reward_psi",) if should else (),
        overlap_ok=True,
    )


def _outcome(*, promoted: bool = False) -> RetrainOutcome:
    return RetrainOutcome(
        promoted=promoted,
        trigger=_trigger(),
        candidate_metrics={"dr": 0.12, "dr_lb": 0.10},
        gate_reason="PROMOTE" if promoted else "HOLD: below gate",
        candidate_model_dir=None,
    )


class _FakeTransport:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list[str], str, str]] = []

    def send(self, *, sender: str, recipients: list[str], subject: str, body: str) -> None:
        self.messages.append((sender, recipients, subject, body))


def test_is_significant_requires_should_retrain_and_breached_count(tmp_path: Path) -> None:
    settings = _settings(tmp_path, alert_min_triggered_signals=1)
    report = _report(reward_triggered=True)
    assert is_significant(report, _trigger(should=True), settings=settings)
    assert not is_significant(report, _trigger(should=False), settings=settings)

    settings2 = _settings(tmp_path, alert_min_triggered_signals=2)
    assert not is_significant(report, _trigger(should=True), settings=settings2)


def test_format_drift_alert_contains_signals_and_verdict() -> None:
    report = _report()
    trigger = _trigger()
    outcome = _outcome(promoted=True)
    subject, body = format_drift_alert(report, trigger, outcome)
    assert "Drift alert" in subject
    assert "reward_psi" in body
    assert "PROMOTE" in body
    assert "0.25" in body or "+0.25" in body


def test_maybe_alert_disabled_by_default(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    decision = maybe_alert_drift(_report(), _trigger(), _outcome(), settings=settings)
    assert decision.sent is False
    assert decision.reason == "disabled"


def test_maybe_alert_sends_once_with_fake_transport(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        alert_email_enabled=True,
        alert_smtp_host="smtp.test",
        alert_email_from="alerts@test",
        alert_email_to="oncall@test",
    )
    transport = _FakeTransport()
    alerter = EmailAlerter(settings=settings, transport=transport)
    state = tmp_path / "alert_state.json"

    d1 = maybe_alert_drift(
        _report(),
        _trigger(),
        _outcome(),
        settings=settings,
        alerter=alerter,
        state_path=state,
        now=datetime(2026, 6, 22, 12, 0, tzinfo=UTC),
    )
    assert d1.sent is True
    assert len(transport.messages) == 1

    d2 = maybe_alert_drift(
        _report(),
        _trigger(),
        _outcome(),
        settings=settings,
        alerter=alerter,
        state_path=state,
        now=datetime(2026, 6, 22, 12, 5, tzinfo=UTC),
    )
    assert d2.sent is False
    assert d2.reason == "debounced"
    assert len(transport.messages) == 1


def test_maybe_alert_dry_run_when_enabled_without_smtp(tmp_path: Path, capsys) -> None:
    settings = _settings(tmp_path, alert_email_enabled=True)
    decision = maybe_alert_drift(
        _report(),
        _trigger(),
        _outcome(),
        settings=settings,
        now=datetime(2026, 6, 22, 12, 0, tzinfo=UTC),
        state_path=tmp_path / "alert_state.json",
    )
    assert decision.reason == "dry_run"
    captured = capsys.readouterr()
    assert "dry-run" in captured.out
    assert "reward_psi" in captured.out


def test_debounce_clears_when_drift_not_significant(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        alert_email_enabled=True,
        alert_smtp_host="smtp.test",
        alert_email_from="a@test",
        alert_email_to="b@test",
        alert_debounce_minutes=30,
    )
    transport = _FakeTransport()
    alerter = EmailAlerter(settings=settings, transport=transport)
    state = tmp_path / "alert_state.json"
    t0 = datetime(2026, 6, 22, 12, 0, tzinfo=UTC)

    maybe_alert_drift(
        _report(),
        _trigger(),
        _outcome(),
        settings=settings,
        alerter=alerter,
        state_path=state,
        now=t0,
    )

    maybe_alert_drift(
        _report(),
        _trigger(should=False),
        _outcome(),
        settings=settings,
        alerter=alerter,
        state_path=state,
        now=t0 + timedelta(minutes=5),
    )

    d3 = maybe_alert_drift(
        _report(),
        _trigger(),
        _outcome(),
        settings=settings,
        alerter=alerter,
        state_path=state,
        now=t0 + timedelta(minutes=10),
    )
    assert d3.sent is True
    assert len(transport.messages) == 2


def test_email_alerter_disabled_returns_disabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    decision = EmailAlerter(settings=settings).send("subj", "body")
    assert decision == AlertDecision(sent=False, reason="disabled")
