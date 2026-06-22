"""Monitor cadence — gate batch monitor/retrain runs on labeled-event volume.

Operators schedule ``run_monitor.py`` / ``run_retrain_loop.py`` on a cron; this
module decides whether enough **new labeled outcomes** have accumulated since the
last :class:`~nba.monitoring.signals.DriftReport` to justify another scoring pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from nba.config import Settings
from nba.monitoring.store_reader import as_utc, latest_drift_report
from nba.schema import BanditEvent


def count_labeled_since(events: list[BanditEvent], since: datetime | None) -> int:
    """Count labeled events logged strictly after ``since``.

    When ``since`` is ``None`` (no prior monitor report), every labeled row counts.
    """
    labeled = [e for e in events if e.reward is not None]
    if since is None:
        return len(labeled)
    cutoff = as_utc(since)
    return sum(1 for e in labeled if as_utc(e.timestamp) > cutoff)


@dataclass(frozen=True)
class MonitorCadence:
    """Whether the monitor batch job is due and the evidence behind it."""

    due: bool
    events_since_last_report: int
    interval: int
    last_report_at: datetime | None


def evaluate_monitor_cadence(events: list[BanditEvent], *, settings: Settings) -> MonitorCadence:
    """Return cadence status using ``settings.monitor_interval_events``."""
    last = latest_drift_report(settings.monitoring_report_path)
    since = last.timestamp if last is not None else None
    n_since = count_labeled_since(events, since)
    interval = settings.monitor_interval_events
    return MonitorCadence(
        due=n_since >= interval,
        events_since_last_report=n_since,
        interval=interval,
        last_report_at=since,
    )


__all__ = ["MonitorCadence", "count_labeled_since", "evaluate_monitor_cadence"]
