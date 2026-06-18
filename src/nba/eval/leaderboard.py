"""An append-only, logged leaderboard of experiments.

Every feature flag (each phase) is measured against the baseline and explicitly judged a **lift**, a
**regression**, or **neutral**. Results are facts: one JSON line is appended per run and prior rows
are never mutated — the same append-only discipline as ``api/store.py``.

Verdict rule (the core ask): an experiment is a **lift** only if the PRIMARY metric
(``realized_shift_value_mean``) rises **and** ``gate_passed`` (the DR lower bound clears the
baseline's DR value by ``ope_min_lift``) — the improvement must be real, not noise. A material drop
in the primary metric (beyond ``ope_min_lift``) is a **regression**; everything else is **neutral**.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from nba.config import Settings
from nba.eval.metrics import ExperimentMetrics

Verdict = Literal["lift", "regression", "neutral"]

#: Metrics where a *lower* value is better; their deltas are sign-flipped so ``+`` always == better.
_LOWER_IS_BETTER = frozenset(
    {"realized_shift_value_std", "decision_regret_mean", "route_time_s_mean"}
)
PRIMARY_METRIC = "realized_shift_value_mean"


@dataclass(frozen=True)
class ExperimentRecord:
    """One graded experiment row."""

    experiment_id: str
    phase: str
    dataset_mode: str
    flags: dict[str, object]
    seeds: list[int]
    metrics: ExperimentMetrics
    baseline_id: str
    deltas: dict[str, float]  # metric -> (this - baseline); sign-normalized so + == better
    gate_passed: bool
    verdict: Verdict
    git_rev: str | None
    timestamp: datetime

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable mapping for one append-only line."""
        return {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "dataset_mode": self.dataset_mode,
            "flags": self.flags,
            "seeds": list(self.seeds),
            "metrics": self.metrics.to_dict(),
            "baseline_id": self.baseline_id,
            "deltas": self.deltas,
            "gate_passed": self.gate_passed,
            "verdict": self.verdict,
            "git_rev": self.git_rev,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> ExperimentRecord:
        """Reconstruct a record from one parsed JSONL line."""
        return cls(
            experiment_id=str(data["experiment_id"]),
            phase=str(data["phase"]),
            dataset_mode=str(data["dataset_mode"]),
            flags=dict(data["flags"]),  # type: ignore[arg-type]
            seeds=list(data["seeds"]),  # type: ignore[arg-type]
            metrics=ExperimentMetrics(**data["metrics"]),  # type: ignore[arg-type]
            baseline_id=str(data["baseline_id"]),
            deltas=dict(data["deltas"]),  # type: ignore[arg-type]
            gate_passed=bool(data["gate_passed"]),
            verdict=data["verdict"],  # type: ignore[assignment]
            git_rev=data["git_rev"],  # type: ignore[assignment]
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
        )


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _signed_delta(metric: str, value: float, baseline: float) -> float:
    raw = value - baseline
    return -raw if metric in _LOWER_IS_BETTER else raw


def _compute_deltas(metrics: ExperimentMetrics, baseline: ExperimentMetrics) -> dict[str, float]:
    deltas: dict[str, float] = {}
    base = baseline.to_dict()
    for metric, value in metrics.to_dict().items():
        bval = base.get(metric)
        if value is None or bval is None:  # e.g. optimality_gap not computed yet
            continue
        deltas[metric] = _signed_delta(metric, float(value), float(bval))
    return deltas


def record_experiment(
    metrics: ExperimentMetrics,
    *,
    settings: Settings,
    experiment_id: str,
    phase: str,
    flags: dict[str, object],
    baseline: ExperimentRecord | None,
) -> ExperimentRecord:
    """Compute deltas vs ``baseline``, derive the verdict, and APPEND one JSONL line.

    When ``baseline`` is ``None`` the row IS the reference (deltas are zero, verdict neutral). Prior
    rows are never modified.
    """
    if baseline is None:
        deltas = {k: 0.0 for k, v in metrics.to_dict().items() if v is not None}
        baseline_value = metrics.ope_value
        baseline_primary = metrics.realized_shift_value_mean
        baseline_id = settings.baseline_experiment_id
    else:
        deltas = _compute_deltas(metrics, baseline.metrics)
        baseline_value = baseline.metrics.ope_value
        baseline_primary = baseline.metrics.realized_shift_value_mean
        baseline_id = baseline.experiment_id

    # The gate quantity: the DR lower confidence bound must clear the baseline DR value by min_lift.
    gate_passed = bool(metrics.ope_lcb > baseline_value + settings.ope_min_lift)

    delta_primary = metrics.realized_shift_value_mean - baseline_primary
    if baseline is None:
        verdict: Verdict = "neutral"
    elif delta_primary > 0.0 and gate_passed:
        verdict = "lift"
    elif delta_primary < -settings.ope_min_lift:
        verdict = "regression"
    else:
        verdict = "neutral"

    record = ExperimentRecord(
        experiment_id=experiment_id,
        phase=phase,
        dataset_mode=settings.dataset_mode,
        flags=flags,
        seeds=list(settings.eval_seeds),
        metrics=metrics,
        baseline_id=baseline_id,
        deltas=deltas,
        gate_passed=gate_passed,
        verdict=verdict,
        git_rev=_git_rev(),
        timestamp=datetime.now(UTC),
    )
    _append(settings.leaderboard_path, record)
    return record


def _append(path: Path, record: ExperimentRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_json()) + "\n")


def load_leaderboard(path: Path) -> list[ExperimentRecord]:
    """Read every appended row (in file order)."""
    if not path.exists():
        return []
    records: list[ExperimentRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(ExperimentRecord.from_json(json.loads(line)))
    return records


def baseline_record(records: list[ExperimentRecord], baseline_id: str) -> ExperimentRecord | None:
    """Return the most recent record whose ``experiment_id`` equals ``baseline_id``."""
    matches = [r for r in records if r.experiment_id == baseline_id]
    return matches[-1] if matches else None


def rank(
    records: list[ExperimentRecord], *, metric: str = PRIMARY_METRIC
) -> list[ExperimentRecord]:
    """Return the latest row per ``experiment_id``, sorted best-first by ``metric``."""
    latest: dict[str, ExperimentRecord] = {}
    for r in records:
        latest[r.experiment_id] = r  # file order => last wins
    return sorted(
        latest.values(),
        key=lambda r: getattr(r.metrics, metric),
        reverse=True,
    )


def render_table(records: list[ExperimentRecord]) -> str:
    """Render a markdown leaderboard, best-first, with verdict, primary delta, and the gate."""
    ranked = rank(records)
    header = (
        "| experiment | phase | dataset | realized value | Δ value | "
        "regret | OPE LCB | gate | verdict |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    rows = [header]
    for r in ranked:
        m = r.metrics
        d = r.deltas.get(PRIMARY_METRIC, 0.0)
        gate = "pass" if r.gate_passed else "fail"
        rows.append(
            f"| {r.experiment_id} | {r.phase} | {r.dataset_mode} | "
            f"{m.realized_shift_value_mean:+.3f} | {d:+.3f} | "
            f"{m.decision_regret_mean:.3f} | {m.ope_lcb:+.3f} | {gate} | {r.verdict} |"
        )
    return "\n".join(rows)
