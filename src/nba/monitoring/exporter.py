"""Prometheus text exposition for the drift monitor.

A read-only layer over the append-only facts the monitor already writes
(``drift_reports.jsonl``, ``retrain_audit.jsonl``, ``deployed.json``) plus
optional EventStore rollups. This module never mutates artifacts and never
imports the simulator oracle.

The exporter is unit-testable without HTTP or Docker: ``build_snapshot`` reads
artifacts into a plain :class:`MonitoringSnapshot`, and
:func:`render_prometheus_text` formats it as Prometheus text. The script
``scripts/run_metrics_exporter.py`` wraps these in a long-lived HTTP server.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nba.api.store import EventStore
from nba.config import Settings
from nba.monitoring.signals import DriftReport, DriftSignal
from nba.monitoring.store_reader import (
    AuditRow,
    DeployedManifest,
    EventStoreRollup,
    count_verdicts,
    event_store_rollups,
    latest_drift_report,
    read_deployed_manifest,
)


@dataclass(frozen=True)
class MonitoringSnapshot:
    """A point-in-time read of every artifact the exporter visualizes."""

    timestamp: datetime
    report: DriftReport | None
    audit_rows: tuple[AuditRow, ...]
    latest_audit: AuditRow | None
    deployed: DeployedManifest | None
    rollups: EventStoreRollup | None
    verdict_counts: dict[str, int] = field(default_factory=dict)


def build_snapshot(
    *,
    settings: Settings,
    store: EventStore | None = None,
    now: datetime | None = None,
) -> MonitoringSnapshot:
    """Read JSONL tails + deployed manifest + optional EventStore rollups."""
    report = latest_drift_report(settings.monitoring_report_path)
    audit_rows = tuple(read_retrain_audit_safe(settings))
    audit_latest = audit_rows[-1] if audit_rows else None
    deployed = read_deployed_manifest(settings.deployed_model_manifest)
    rollups = event_store_rollups(store, recent_window=settings.monitor_recent_window)
    counts = count_verdicts(audit_rows)
    return MonitoringSnapshot(
        timestamp=now or datetime.now(UTC),
        report=report,
        audit_rows=audit_rows,
        latest_audit=audit_latest,
        deployed=deployed,
        rollups=rollups,
        verdict_counts=counts,
    )


def read_retrain_audit_safe(settings: Settings) -> list[AuditRow]:
    """Local import-safe wrapper around :func:`store_reader.read_retrain_audit`."""
    from nba.monitoring.store_reader import read_retrain_audit  # noqa: PLC0415

    return read_retrain_audit(settings.retrain_audit_path)


def _fmt_label(name: str, value: str) -> str:
    """Format a single ``label="value"`` pair (escapes ``"`` and ``\\``)."""
    safe = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{name}="{safe}"'


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(_fmt_label(k, v) for k, v in labels.items()) + "}"


@dataclass(frozen=True)
class _MetricSpec:
    """One Prometheus gauge spec to render."""

    name: str
    help: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)


def _signal_value(report: DriftReport | None, name: str) -> float | None:
    if report is None:
        return None
    try:
        return report.signal(name).value
    except KeyError:
        return None


def _overlap_min_p_from_report(report: DriftReport | None) -> float | None:
    """Parse ``min_p`` from the overlap_health signal detail when EventStore rollups are absent."""
    if report is None:
        return None
    try:
        detail = report.signal("overlap_health").detail
    except KeyError:
        return None
    match = re.search(r"min_p=([0-9.]+)", detail)
    return float(match.group(1)) if match else None


def _signal_threshold(settings: Settings, name: str) -> float | None:
    """Look up the configured threshold for a signal name."""
    mapping = {
        "reward_psi": settings.drift_reward_psi_threshold,
        "calibration_drift": settings.drift_calibration_delta_threshold,
        "feature_psi_max": settings.drift_feature_psi_threshold,
        "rolling_dr_drop": settings.drift_rolling_dr_drop_threshold,
        "overlap_health": min(settings.drift_min_propensity_floor, settings.drift_min_ess_fraction),
    }
    return mapping.get(name)


def _spec_rows(snapshot: MonitoringSnapshot, settings: Settings) -> list[_MetricSpec]:
    """Build the full list of metric specs from the snapshot + settings."""
    specs: list[_MetricSpec] = []
    report = snapshot.report

    # Per-signal gauges + threshold companion metrics.
    signal_names = (
        "reward_psi",
        "calibration_drift",
        "feature_psi_max",
        "rolling_dr_drop",
        "overlap_health",
    )
    for name in signal_names:
        val = _signal_value(report, name)
        if val is not None:
            specs.append(
                _MetricSpec(
                    name=f"nba_drift_{name}",
                    help=f"Latest value of the {name} drift signal",
                    value=float(val),
                )
            )
            # Threshold companion metric — same label-free form so Grafana can join on name.
            threshold = _signal_threshold(settings, name)
            if threshold is not None:
                specs.append(
                    _MetricSpec(
                        name=f"nba_drift_{name}_threshold",
                        help=f"Configured threshold for {name}",
                        value=float(threshold),
                    )
                )
            # 0/1 triggered flag with a signal label for annotation queries.
            try:
                triggered = int(report.signal(name).triggered) if report else 0
            except KeyError:
                triggered = 0
            specs.append(
                _MetricSpec(
                    name="nba_drift_signal_triggered",
                    help="1 when the signal is currently triggered, else 0",
                    value=float(triggered),
                    labels={"signal": name},
                )
            )

    # Calibration recent/delta explicit gauges (mirror doc table).
    if report is not None:
        try:
            calib = report.signal("calibration_drift")
            specs.append(
                _MetricSpec(
                    name="nba_drift_calibration_mae_delta",
                    help="Δ calibration MAE (recent − reference)",
                    value=float(calib.value),
                )
            )
        except KeyError:
            pass

    # Deployed model metrics.
    if snapshot.deployed is not None:
        deployed = snapshot.deployed
        specs.append(
            _MetricSpec(
                name="nba_deployed_dr_lb",
                help="DR lower bound of the currently deployed model/policy",
                value=float(deployed.dr_lower_bound),
            )
        )
        age_days = (snapshot.timestamp - deployed.promoted_at).total_seconds() / 86400.0
        specs.append(
            _MetricSpec(
                name="nba_deployed_model_age_days",
                help="Days since the deployed model was promoted",
                value=float(max(0.0, age_days)),
            )
        )

    # EventStore rollups (primary source for event/reward/overlap panels).
    if snapshot.rollups is not None:
        specs.append(
            _MetricSpec(
                name="nba_events_labeled_total",
                help="Total labeled events in the EventStore",
                value=float(snapshot.rollups.n_labeled),
            )
        )
        specs.append(
            _MetricSpec(
                name="nba_events_recent_mean_reward",
                help=f"Mean reward over the last {settings.monitor_recent_window} labeled events",
                value=float(snapshot.rollups.recent_mean_reward),
            )
        )
        overlap_min_p = snapshot.rollups.recent_min_propensity
    else:
        overlap_min_p = _overlap_min_p_from_report(report)

    if overlap_min_p is not None:
        specs.append(
            _MetricSpec(
                name="nba_drift_overlap_min_propensity",
                help="Min propensity over the recent labeled window",
                value=float(overlap_min_p),
            )
        )

    # Retrain audit counters — always emit promote/hold so Grafana stat panels never show "No data".
    for verdict in ("promote", "hold"):
        specs.append(
            _MetricSpec(
                name="nba_retrain_total",
                help="Total retrain audit rows by verdict",
                value=float(snapshot.verdict_counts.get(verdict, 0)),
                labels={"verdict": verdict},
            )
        )

    return specs


def render_prometheus_text(snapshot: MonitoringSnapshot, *, settings: Settings) -> str:
    """Render the snapshot as Prometheus text exposition format."""
    specs = _spec_rows(snapshot, settings)
    # Deduplicate by (name, labels) keeping the last definition's help/type to avoid dup HELP.
    seen: dict[tuple[str, str], _MetricSpec] = {}
    type_emitted: set[str] = set()
    help_emitted: set[str] = set()
    out: list[str] = []
    for spec in specs:
        key = (spec.name, _fmt_labels(spec.labels))
        # Emit # HELP and # TYPE only on first occurrence of a name.
        if spec.name not in help_emitted:
            out.append(f"# HELP {spec.name} {spec.help}")
            help_emitted.add(spec.name)
        if spec.name not in type_emitted:
            out.append(f"# TYPE {spec.name} gauge")
            type_emitted.add(spec.name)
        out.append(f"{spec.name}{_fmt_labels(spec.labels)} {_format_value(spec.value)}")
        seen[key] = spec
    return "\n".join(out) + ("\n" if out else "")


def _format_value(v: float) -> str:
    """Prometheus float formatting — avoid scientific notation for small numbers."""
    if v == int(v) and abs(v) < 1e15:
        return f"{v:.1f}"
    return f"{v:.6f}"


__all__ = [
    "MonitoringSnapshot",
    "build_snapshot",
    "render_prometheus_text",
]


# Silence unused-import warnings for re-exports kept in the public surface.
_: tuple[type, ...] = (DriftSignal, Sequence)
