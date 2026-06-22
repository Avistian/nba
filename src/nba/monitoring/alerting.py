"""Email alerting on significant drift — notification side-channel only.

Drift is detected by :mod:`nba.monitoring.signals` and
:mod:`nba.monitoring.triggers`. This module sends a debounced email (or
dry-run print) when a scored report is **significant**. It is intentionally
kept out of :class:`~nba.monitoring.retrain.RetrainLoop` so core retrain
logic stays network-free and unit-testable.

Scripts call :func:`maybe_alert_drift` after ``RetrainLoop.run`` returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from nba.config import Settings
from nba.monitoring.retrain import RetrainOutcome
from nba.monitoring.signals import DriftReport
from nba.monitoring.triggers import RetrainTrigger

# Primary drift signals (overlap_health warns but does not alone trigger retrain).
_PRIMARY_SIGNALS = frozenset(
    {"reward_psi", "calibration_drift", "feature_psi_max", "rolling_dr_drop"}
)


class EmailTransport(Protocol):
    """Injectable mail sender for tests (no real socket)."""

    def send(self, *, sender: str, recipients: list[str], subject: str, body: str) -> None: ...


@dataclass(frozen=True)
class AlertDecision:
    """Why an alert was sent, skipped, or dry-run printed."""

    sent: bool
    reason: str  # e.g. "sent", "disabled", "not_significant", "debounced", "dry_run"


def _alert_state_path(settings: Settings, state_path: Path | None = None) -> Path:
    if state_path is not None:
        return state_path
    return settings.monitoring_report_path.parent / "alert_state.json"


def _count_breached_primary(report: DriftReport) -> int:
    return sum(1 for s in report.signals if s.name in _PRIMARY_SIGNALS and s.triggered)


def is_significant(
    report: DriftReport,
    trigger: RetrainTrigger,
    *,
    settings: Settings,
) -> bool:
    """Return whether the report warrants an email alert."""
    if not trigger.should_retrain:
        return False
    breached = _count_breached_primary(report)
    return breached >= settings.alert_min_triggered_signals


def format_drift_alert(
    report: DriftReport,
    trigger: RetrainTrigger,
    outcome: RetrainOutcome | None,
) -> tuple[str, str]:
    """Return ``(subject, body)`` for a significant-drift alert."""
    breached = [s for s in report.signals if s.name in _PRIMARY_SIGNALS and s.triggered]
    lines = [
        "NBA drift monitor: significant drift detected",
        "",
        f"Timestamp: {report.timestamp.isoformat()}",
        f"Overlap OK: {report.overlap_ok}",
        f"Trigger reasons: {', '.join(trigger.reasons) or 'none'}",
        "",
        "Breached signals:",
    ]
    for sig in breached:
        lines.append(
            f"  - {sig.name}: {sig.value:+.4f} (threshold {sig.threshold:.3f}) — {sig.detail}"
        )
    if not breached:
        lines.append("  (none — scheduled trigger only)")

    lines.append("")
    if outcome is not None:
        verdict = "PROMOTE" if outcome.promoted else "HOLD"
        lines.append(f"Retrain verdict: {verdict}")
        lines.append(f"Gate reason: {outcome.gate_reason}")
        if outcome.candidate_metrics:
            dr = outcome.candidate_metrics.get("dr")
            dr_lb = outcome.candidate_metrics.get("dr_lb")
            if dr is not None and dr_lb is not None:
                lines.append(f"Candidate DR: {dr:+.4f} (lb={dr_lb:+.4f})")
    else:
        lines.append("Retrain verdict: (not run)")

    subject = f"[NBA] Drift alert — {len(breached)} signal(s) breached"
    return subject, "\n".join(lines)


def _parse_recipients(raw: str) -> list[str]:
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def _has_smtp_config(settings: Settings) -> bool:
    return bool(settings.alert_smtp_host.strip() and settings.alert_email_to.strip())


class EmailAlerter:
    """Thin SMTP sender; accepts an injectable :class:`EmailTransport`."""

    def __init__(
        self,
        *,
        settings: Settings,
        transport: EmailTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    def send(self, subject: str, body: str) -> AlertDecision:
        """Send the alert email, dry-run print, or skip."""
        if not self._settings.alert_email_enabled:
            return AlertDecision(sent=False, reason="disabled")

        recipients = _parse_recipients(self._settings.alert_email_to)
        sender = self._settings.alert_email_from.strip() or self._settings.alert_smtp_user.strip()

        if not _has_smtp_config(self._settings):
            print(f"\n--- NBA drift alert (dry-run) ---\nSubject: {subject}\n\n{body}\n---\n")
            return AlertDecision(sent=False, reason="dry_run")

        if not sender:
            return AlertDecision(sent=False, reason="missing_sender")

        if self._transport is not None:
            self._transport.send(
                sender=sender, recipients=recipients, subject=subject, body=body
            )
        else:
            self._send_smtp(sender=sender, recipients=recipients, subject=subject, body=body)

        return AlertDecision(sent=True, reason="sent")

    def _send_smtp(
        self, *, sender: str, recipients: list[str], subject: str, body: str
    ) -> None:
        import smtplib  # noqa: PLC0415 — lazy import; only when actually sending

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        msg.set_content(body)

        settings = self._settings
        if settings.alert_smtp_use_tls:
            with smtplib.SMTP(settings.alert_smtp_host, settings.alert_smtp_port) as smtp:
                smtp.starttls()
                if settings.alert_smtp_user:
                    smtp.login(settings.alert_smtp_user, settings.alert_smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP_SSL(settings.alert_smtp_host, settings.alert_smtp_port) as smtp:
                if settings.alert_smtp_user:
                    smtp.login(settings.alert_smtp_user, settings.alert_smtp_password)
                smtp.send_message(msg)


def _load_alert_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_alert_state(path: Path, *, last_sent_at: datetime | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if last_sent_at is not None:
        payload["last_sent_at"] = last_sent_at.isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _debounced(
    *,
    settings: Settings,
    state_path: Path,
    now: datetime,
) -> bool:
    """Return True when a prior alert was sent within the debounce window."""
    state = _load_alert_state(state_path)
    raw = state.get("last_sent_at")
    if not isinstance(raw, str):
        return False
    try:
        last = datetime.fromisoformat(raw)
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
    except ValueError:
        return False
    elapsed_min = (now - last).total_seconds() / 60.0
    return elapsed_min < settings.alert_debounce_minutes


def maybe_alert_drift(
    report: DriftReport | None,
    trigger: RetrainTrigger,
    outcome: RetrainOutcome | None,
    *,
    settings: Settings,
    alerter: EmailAlerter | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> AlertDecision:
    """Send (or dry-run print) a debounced email when drift is significant."""
    if report is None:
        return AlertDecision(sent=False, reason="no_report")

    now_utc = now or datetime.now(UTC)
    state_file = _alert_state_path(settings, state_path)

    if not is_significant(report, trigger, settings=settings):
        # Drift cleared — reset debounce so the next episode can alert immediately.
        _save_alert_state(state_file, last_sent_at=None)
        return AlertDecision(sent=False, reason="not_significant")

    if _debounced(settings=settings, state_path=state_file, now=now_utc):
        return AlertDecision(sent=False, reason="debounced")

    subject, body = format_drift_alert(report, trigger, outcome)
    mailer = alerter or EmailAlerter(settings=settings)
    decision = mailer.send(subject, body)

    if decision.sent or decision.reason == "dry_run":
        _save_alert_state(state_file, last_sent_at=now_utc)

    return decision


__all__ = [
    "AlertDecision",
    "EmailAlerter",
    "EmailTransport",
    "format_drift_alert",
    "is_significant",
    "maybe_alert_drift",
]
