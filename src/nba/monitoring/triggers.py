"""Trigger evaluation — decide whether the monitor mandates a retrain.

Rule (default, all flags off by default — ``use_drift_monitoring`` gates the
whole batch job):

    retrain_triggered =
        (reward_psi > threshold)
        OR (calibration_drift triggered)
        OR (feature_psi_max > threshold)
        OR (rolling_dr_drop > threshold)
        OR (scheduled: days_since_promote >= retrain_max_age_days AND
            n_new_labeled >= retrain_min_new_events)

Overlap failures **warn** and block promotion until resolved; they do NOT alone
trigger a retrain (retraining on bad logs makes things worse). When overlap is
bad the trigger returns ``should_retrain=False`` with ``reasons=("overlap_bad",)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from nba.config import Settings
from nba.monitoring.signals import DriftReport


@dataclass(frozen=True)
class RetrainTrigger:
    """The monitor's verdict on whether to retrain + why."""

    should_retrain: bool
    reasons: tuple[str, ...]  # e.g. ("reward_psi", "scheduled_max_age")
    overlap_ok: bool


def evaluate_triggers(
    report: DriftReport, *, settings: Settings, days_since_promote: float, n_new: int
) -> RetrainTrigger:
    """Evaluate the trigger rule over a scored :class:`DriftReport`."""
    # Overlap failure short-circuits: do not retrain on OPE-invalid logs.
    if not report.overlap_ok:
        return RetrainTrigger(should_retrain=False, reasons=("overlap_bad",), overlap_ok=False)

    reasons: list[str] = []
    for name in ("reward_psi", "feature_psi_max", "rolling_dr_drop"):
        sig = report.signal(name)
        if sig.triggered:
            reasons.append(name)
    calib = report.signal("calibration_drift")
    if calib.triggered:
        reasons.append("calibration_drift")

    # Scheduled safety ceiling — only if enough new data has accumulated.
    scheduled = (
        days_since_promote >= settings.retrain_max_age_days
        and n_new >= settings.retrain_min_new_events
    )
    if scheduled:
        reasons.append("scheduled_max_age")

    should = bool(reasons)
    return RetrainTrigger(should_retrain=should, reasons=tuple(reasons), overlap_ok=True)


__all__ = ["RetrainTrigger", "evaluate_triggers"]
