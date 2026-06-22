"""Read-only access to the drift JSONL / retrain audit / deployed manifest / EventStore.

These helpers never mutate artifacts. They are the sole data source for the
Prometheus exporter and for any future read-only dashboard backends.

The functions tolerate missing files (return empty/None) so the exporter can
run before any monitor/retrain has produced artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from nba.api.store import EventStore
from nba.monitoring.signals import DriftReport


@dataclass(frozen=True)
class AuditRow:
    """One retrain audit row — verdict, trigger reasons, metrics, timestamp."""

    timestamp: datetime
    verdict: str  # "promote" | "hold"
    reasons: tuple[str, ...]
    promoted: bool
    candidate_dr: float | None
    candidate_dr_lb: float | None
    deployed_dr: float | None
    overlap_ok: bool

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable mapping for one append-only line."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "promoted": self.promoted,
            "candidate_dr": self.candidate_dr,
            "candidate_dr_lb": self.candidate_dr_lb,
            "deployed_dr": self.deployed_dr,
            "overlap_ok": self.overlap_ok,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> AuditRow:
        """Reconstruct an audit row from one parsed JSONL line."""
        return cls(
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
            verdict=str(data["verdict"]),
            reasons=tuple(str(r) for r in data.get("reasons", [])),  # type: ignore[arg-type]
            promoted=bool(data["promoted"]),
            candidate_dr=_maybe_float(data.get("candidate_dr")),
            candidate_dr_lb=_maybe_float(data.get("candidate_dr_lb")),
            deployed_dr=_maybe_float(data.get("deployed_dr")),
            overlap_ok=bool(data.get("overlap_ok", True)),
        )


@dataclass(frozen=True)
class DeployedManifest:
    """The ``deployed.json`` manifest — points at the active model dir + metrics."""

    model_dir: str
    promoted_at: datetime
    dr_value: float
    dr_lower_bound: float
    baseline_value: float
    feature_names: list[str]

    @classmethod
    def from_json(cls, data: dict[str, object]) -> DeployedManifest:
        """Reconstruct a manifest from a parsed ``deployed.json`` mapping."""
        return cls(
            model_dir=str(data["model_dir"]),
            promoted_at=datetime.fromisoformat(str(data["promoted_at"])),
            dr_value=float(data["dr_value"]),  # type: ignore[arg-type]
            dr_lower_bound=float(data["dr_lower_bound"]),  # type: ignore[arg-type]
            baseline_value=float(data["baseline_value"]),  # type: ignore[arg-type]
            feature_names=list(data.get("feature_names", [])),  # type: ignore[arg-type]
        )


def _maybe_float(v: object) -> float | None:
    if v is None:
        return None
    return float(v)  # type: ignore[arg-type]


def read_drift_reports(path: Path | str) -> list[DriftReport]:
    """Read every report from ``drift_reports.jsonl`` (file order)."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[DriftReport] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(DriftReport.from_json(json.loads(line)))
    return out


def read_retrain_audit(path: Path | str) -> list[AuditRow]:
    """Read every audit row from ``retrain_audit.jsonl`` (file order)."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[AuditRow] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(AuditRow.from_json(json.loads(line)))
    return out


def read_deployed_manifest(path: Path | str) -> DeployedManifest | None:
    """Read the deployed manifest. Returns ``None`` if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    return DeployedManifest.from_json(json.loads(p.read_text(encoding="utf-8")))


def latest_drift_report(path: Path | str) -> DriftReport | None:
    """Return the most recent :class:`DriftReport` (or ``None`` when absent)."""
    reports = read_drift_reports(path)
    return reports[-1] if reports else None


def latest_audit(path: Path | str) -> AuditRow | None:
    """Return the most recent :class:`AuditRow` (or ``None``)."""
    rows = read_retrain_audit(path)
    return rows[-1] if rows else None


@dataclass(frozen=True)
class EventStoreRollup:
    """Lightweight aggregates over the EventStore for exporter gauges."""

    n_labeled: int
    recent_mean_reward: float
    recent_min_propensity: float


def event_store_rollups(
    store: EventStore | None, *, recent_window: int = 2000
) -> EventStoreRollup | None:
    """Aggregate labeled events from ``store``.

    Returns ``None`` when ``store`` is ``None`` (exporter may run with no DB).
    """
    if store is None:
        return None
    events = store.load_events()
    labeled = [e for e in events if e.reward is not None]
    if not labeled:
        return EventStoreRollup(n_labeled=0, recent_mean_reward=0.0, recent_min_propensity=0.0)
    recent = labeled[-recent_window:]
    rewards = [float(e.reward) for e in recent if e.reward is not None]
    propensities = [float(e.propensity) for e in recent]
    return EventStoreRollup(
        n_labeled=len(labeled),
        recent_mean_reward=float(sum(rewards) / len(recent)),
        recent_min_propensity=float(min(propensities)),
    )


def count_verdicts(rows: Sequence[AuditRow]) -> dict[str, int]:
    """Return counts of each verdict across audit rows (e.g. promote=3, hold=1)."""
    out: dict[str, int] = {}
    for r in rows:
        out[r.verdict] = out.get(r.verdict, 0) + 1
    return out


__all__ = [
    "AuditRow",
    "DeployedManifest",
    "EventStoreRollup",
    "count_verdicts",
    "event_store_rollups",
    "latest_audit",
    "latest_drift_report",
    "read_deployed_manifest",
    "read_drift_reports",
    "read_retrain_audit",
]
