"""End-to-end drift demo: frozen model degrades → monitor fires → retrain recovers.

Narrative (seeded, writes ``artifacts/drift_demo_report.json``):

1. Generate **pre-drift** logs; fit + gate the deployed model; record baseline DR/regret.
2. Generate **post-drift** logs (``DriftSpec(at_fraction=0)`` — entirely post-change world).
3. Simulate K shifts serving with the **frozen** deployed model:
   - Track rolling calibration MAE, realized reward, regret vs oracle.
   - Run the monitor after each shift → expect signals to trigger mid-run.
4. Run the retrain loop once triggered → candidate must pass the DR gate to promote.
5. Simulate K more shifts with the promoted model → regret/reward recover toward baseline.
6. Write ``artifacts/drift_demo_report.json`` (drift signals per shift, regret pre/during/post
   retrain).

CLI:
    uv run python scripts/simulate_drift_demo.py --n-pre 15000 --n-post 8000 --shifts 6 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Allow reuse of run_demo helpers (oracle, dense block) — same pattern as eval/metrics.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nba.config import Settings  # noqa: E402
from nba.data.drift import DriftSpec, generate_logs_with_drift  # noqa: E402
from nba.data.simulator import generate_logs  # noqa: E402
from nba.eval.oracle import oracle_for  # noqa: E402
from nba.monitoring.retrain import (  # noqa: E402
    RetrainLoop,
    _split_windows,
    bootstrap_deployed,
    load_deployed_stack,
)
from nba.monitoring.signals import (  # noqa: E402
    DriftReportContext,
    append_report,
    build_drift_report,
)
from nba.monitoring.store_reader import read_deployed_manifest  # noqa: E402
from nba.ope.estimators import LoggedBatch  # noqa: E402
from nba.ope.gate import PromotionGate  # noqa: E402
from nba.reward.model import RewardModel  # noqa: E402

# Isolated artifact tree for the drift demo — never the shared production defaults.
_DRIFT_DEMO_ROOT = Path("artifacts/drift_demo")
_DRIFT_DEMO_DB_PATH = _DRIFT_DEMO_ROOT / "events.db"
_DRIFT_DEMO_MODEL_DIR = _DRIFT_DEMO_ROOT / "models"
_DRIFT_DEMO_DEPLOYED_MANIFEST = _DRIFT_DEMO_MODEL_DIR / "deployed.json"
_DRIFT_DEMO_MONITORING_REPORT = _DRIFT_DEMO_ROOT / "monitoring" / "drift_reports.jsonl"
_DRIFT_DEMO_RETRAIN_AUDIT = _DRIFT_DEMO_ROOT / "monitoring" / "retrain_audit.jsonl"

_DEFAULT_PRODUCTION_DB_PATH = Path("artifacts/events.db")
_DEFAULT_PRODUCTION_MODEL_DIR = Path("artifacts/models")
_DEFAULT_PRODUCTION_DEPLOYED_MANIFEST = Path("artifacts/models/deployed.json")
_DEFAULT_PRODUCTION_MONITORING_REPORT = Path("artifacts/monitoring/drift_reports.jsonl")
_DEFAULT_PRODUCTION_RETRAIN_AUDIT = Path("artifacts/monitoring/retrain_audit.jsonl")


def _redirect_production_path(value: Path, *, production: Path, demo: Path) -> Path:
    """Return ``demo`` when ``value`` is the shared production default."""
    if value.resolve() == production.resolve():
        return demo
    return value


def _demo_db_path(settings: Settings) -> Path:
    """Return a demo-safe db path, redirecting away from the shared production store."""
    return _redirect_production_path(
        settings.db_path,
        production=_DEFAULT_PRODUCTION_DB_PATH,
        demo=_DRIFT_DEMO_DB_PATH,
    )


def _demo_settings(settings: Settings) -> Settings:
    """Redirect production artifact defaults to the isolated drift-demo tree."""
    return settings.model_copy(
        update={
            "db_path": _demo_db_path(settings),
            "model_dir": _redirect_production_path(
                settings.model_dir,
                production=_DEFAULT_PRODUCTION_MODEL_DIR,
                demo=_DRIFT_DEMO_MODEL_DIR,
            ),
            "deployed_model_manifest": _redirect_production_path(
                settings.deployed_model_manifest,
                production=_DEFAULT_PRODUCTION_DEPLOYED_MANIFEST,
                demo=_DRIFT_DEMO_DEPLOYED_MANIFEST,
            ),
            "monitoring_report_path": _redirect_production_path(
                settings.monitoring_report_path,
                production=_DEFAULT_PRODUCTION_MONITORING_REPORT,
                demo=_DRIFT_DEMO_MONITORING_REPORT,
            ),
            "retrain_audit_path": _redirect_production_path(
                settings.retrain_audit_path,
                production=_DEFAULT_PRODUCTION_RETRAIN_AUDIT,
                demo=_DRIFT_DEMO_RETRAIN_AUDIT,
            ),
        }
    )


def _is_drift_demo_db(db_path: Path) -> bool:
    return db_path.resolve() == _DRIFT_DEMO_DB_PATH.resolve()


@dataclass
class ShiftRecord:
    """One shift's signals + realized/regret metrics."""

    shift_index: int
    phase: str  # "pre" | "frozen" | "post_retrain"
    n_events: int
    mean_reward: float
    mean_regret: float
    calibration_mae: float
    signals: dict[str, float]
    overlap_ok: bool


@dataclass
class DriftDemoReport:
    """The structured drift demo outcome."""

    seed: int
    n_pre: int
    n_post: int
    shifts: int
    baseline_dr: float
    promoted: bool
    promote_shift: int | None
    shift_records: list[ShiftRecord] = field(default_factory=list)

    def to_json(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "shifts": self.shifts,
            "baseline_dr": self.baseline_dr,
            "promoted": self.promoted,
            "promote_shift": self.promote_shift,
            "shift_records": [asdict(s) for s in self.shift_records],
        }


def _mean_regret(events, oracle) -> float:
    if not events:
        return 0.0
    regrets = []
    for e in events:
        chosen = oracle.true_reward(e.context, e.action)
        best = oracle.true_reward(e.context, oracle.true_best_action(e.context))
        regrets.append(best - chosen)
    return float(np.mean(regrets))


def _calib_mae(model: RewardModel, events) -> float:
    if not events:
        return 0.0
    errs = [abs(model.q(e.context, e.action) - (e.reward or 0.0)) for e in events]
    return float(np.mean(errs))


def _load_promoted_stack(
    *,
    settings: Settings,
    events: list,
) -> tuple[RewardModel, object, float]:
    """Reload model + policy after promotion so drift scoring uses an aligned stack."""
    deployed_model, deployed_policy, _manifest, baseline_dr = load_deployed_stack(
        settings=settings, events=events
    )
    return deployed_model, deployed_policy, baseline_dr


def _signals_from_report(report) -> tuple[dict[str, float], bool]:
    """Extract shift-record signal values from a scored DriftReport."""
    return {s.name: s.value for s in report.signals}, report.overlap_ok


def _score_drift(
    *,
    model,
    policy,
    events,
    settings,
    deployed_dr,
    promoted_at=None,
):
    """Score drift with the same reference/recent windows as ``RetrainLoop.run``."""
    reference_events, recent_events = _split_windows(
        events, settings=settings, promoted_at=promoted_at
    )
    ref = LoggedBatch.from_events(reference_events)
    recent = LoggedBatch.from_events(recent_events)
    report = build_drift_report(
        ctx=DriftReportContext(
            model=model,
            policy=policy,
            reference=ref,
            recent=recent,
            deployed_dr=deployed_dr,
        ),
        settings=settings,
    )
    append_report(report, settings.monitoring_report_path)
    return _signals_from_report(report)


def _persist_events(settings: Settings, events: list) -> int:
    """Write labeled events to the EventStore so exporter rollups populate Grafana panels."""
    from nba.api.store import EventStore  # noqa: PLC0415

    labeled = [e for e in events if e.reward is not None and e.outcome is not None]
    if not labeled:
        return 0
    db_path = _demo_db_path(settings)
    # Reset only the isolated demo store; production logs are append-only and must never be wiped.
    if _is_drift_demo_db(db_path) and db_path.exists():
        db_path.unlink()
    store = EventStore(db_path)
    try:
        return store.ingest_bandit_events(labeled, policy_name="drift_demo")
    finally:
        store.close()


def run_drift_demo(
    *,
    n_pre: int = 15_000,
    n_post: int = 8_000,
    shifts: int = 6,
    seed: int = 7,
    settings: Settings | None = None,
    report_path: Path | None = Path("artifacts/drift_demo_report.json"),
) -> DriftDemoReport:
    """Run the full drift demo narrative and write the report JSON."""
    settings = _demo_settings((settings or Settings()).model_copy(update={"seed": seed}))
    settings.ensure_dirs()
    np.random.default_rng(seed)

    # 1. Pre-drift logs → bootstrap deployed model + manifest.
    pre_events = generate_logs(n_pre, settings=settings, seed=seed)
    deployed_model, deployed_policy, manifest, baseline_dr = bootstrap_deployed(
        settings=settings, events=pre_events
    )
    oracle = oracle_for(settings)

    report = DriftDemoReport(
        seed=seed,
        n_pre=n_pre,
        n_post=n_post,
        shifts=shifts,
        baseline_dr=baseline_dr,
        promoted=False,
        promote_shift=None,
    )

    # 2. Generate one batch of post-drift logs we will slice into K "shifts".
    spec = DriftSpec(at_fraction=0.0, reward_scale=1.3, knock_evening_boost=0.15)
    post_events_all = generate_logs_with_drift(
        n_post * shifts, settings=settings, seed=seed + 100, spec=spec
    )
    per_shift = max(1, len(post_events_all) // shifts)
    post_shifts = [post_events_all[i * per_shift : (i + 1) * per_shift] for i in range(shifts)]

    # 3. Serve K shifts with the frozen deployed model; monitor after each.
    gate = PromotionGate(z=settings.ope_z, min_lift=settings.ope_min_lift)
    loop = RetrainLoop(settings=settings, gate=gate)
    promote_shift: int | None = None

    for k in range(shifts):
        shift_events = post_shifts[k]
        events_cumulative = list(pre_events) + [e for s in post_shifts[: k + 1] for e in s]
        calib = _calib_mae(deployed_model, shift_events)
        regret = _mean_regret(shift_events, oracle)
        mean_reward = float(np.mean([e.reward for e in shift_events if e.reward is not None]))

        # Conditional retrain: RetrainLoop is the sole JSONL writer per frozen shift
        # (matches one production monitor pass).
        if not report.promoted:
            outcome = loop.run(
                deployed_model=deployed_model,
                deployed_policy=deployed_policy,
                events=events_cumulative,
                deployed_dr=baseline_dr,
            )
            drift_report = outcome.report
            if drift_report is not None:
                signals, overlap_ok = _signals_from_report(drift_report)
            else:
                manifest = read_deployed_manifest(settings.deployed_model_manifest)
                promoted_at = manifest.promoted_at if manifest else None
                signals, overlap_ok = _score_drift(
                    model=deployed_model,
                    policy=deployed_policy,
                    events=events_cumulative,
                    settings=settings,
                    deployed_dr=baseline_dr,
                    promoted_at=promoted_at,
                )
            if outcome.promoted and outcome.candidate_model_dir is not None:
                deployed_model, deployed_policy, baseline_dr = _load_promoted_stack(
                    settings=settings,
                    events=events_cumulative,
                )
                report.promoted = True
                promote_shift = k
        else:
            manifest = read_deployed_manifest(settings.deployed_model_manifest)
            promoted_at = manifest.promoted_at if manifest else None
            signals, overlap_ok = _score_drift(
                model=deployed_model,
                policy=deployed_policy,
                events=events_cumulative,
                settings=settings,
                deployed_dr=baseline_dr,
                promoted_at=promoted_at,
            )

        report.shift_records.append(
            ShiftRecord(
                shift_index=k,
                phase="frozen",
                n_events=len(shift_events),
                mean_reward=mean_reward,
                mean_regret=regret,
                calibration_mae=calib,
                signals=signals,
                overlap_ok=overlap_ok,
            )
        )

    # 4. Serve K more shifts only after a successful promote (not on HOLD / no trigger).
    more_post: list = []
    if report.promoted:
        more_post = generate_logs_with_drift(
            n_post * shifts, settings=settings, seed=seed + 200, spec=spec
        )
        more_shifts = [more_post[i * per_shift : (i + 1) * per_shift] for i in range(shifts)]
        frozen_events = list(pre_events) + list(post_events_all)
        manifest = read_deployed_manifest(settings.deployed_model_manifest)
        promoted_at = manifest.promoted_at if manifest else None
        for k, shift_events in enumerate(more_shifts):
            events_cumulative = frozen_events + [e for s in more_shifts[: k + 1] for e in s]
            signals, overlap_ok = _score_drift(
                model=deployed_model,
                policy=deployed_policy,
                events=events_cumulative,
                settings=settings,
                deployed_dr=baseline_dr,
                promoted_at=promoted_at,
            )
            calib = _calib_mae(deployed_model, shift_events)
            regret = _mean_regret(shift_events, oracle)
            mean_reward = float(np.mean([e.reward for e in shift_events if e.reward is not None]))
            report.shift_records.append(
                ShiftRecord(
                    shift_index=shifts + k,
                    phase="post_retrain",
                    n_events=len(shift_events),
                    mean_reward=mean_reward,
                    mean_regret=regret,
                    calibration_mae=calib,
                    signals=signals,
                    overlap_ok=overlap_ok,
                )
            )

    report.promote_shift = promote_shift

    # Seed EventStore so Grafana panels for labeled events / mean reward / min_p populate.
    all_events = list(pre_events) + list(post_events_all) + list(more_post)
    n_ingested = _persist_events(settings, all_events)
    if n_ingested:
        print(f"ingested {n_ingested:,} labeled events into {settings.db_path}")

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    return report


def _print_report(report: DriftDemoReport) -> None:
    print(f"\n{'=' * 70}")
    print(
        f"NBA drift demo  (seed={report.seed}, n_pre={report.n_pre:,}, "
        f"n_post={report.n_post:,}, shifts={report.shifts})"
    )
    print("=" * 70)
    print(f"baseline DR: {report.baseline_dr:+.4f}")
    print(f"promoted: {report.promoted}  (at shift {report.promote_shift})\n")
    print(
        f"{'idx':>3}  {'phase':<12} {'mean_r':>8} {'regret':>8} {'calib':>8}  "
        f"{'reward_psi':>11}  {'overlap':>7}"
    )
    for s in report.shift_records:
        print(
            f"{s.shift_index:>3}  {s.phase:<12} {s.mean_reward:>+8.4f} "
            f"{s.mean_regret:>8.4f} {s.calibration_mae:>8.4f}  "
            f"{s.signals.get('reward_psi', 0.0):>11.4f}  "
            f"{'ok' if s.overlap_ok else 'BAD':>7}"
        )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a drift -> monitor -> retrain -> recovery demo."
    )
    parser.add_argument("--n-pre", type=int, default=15_000)
    parser.add_argument("--n-post", type=int, default=8_000)
    parser.add_argument("--shifts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/drift_demo_report.json"))
    args = parser.parse_args()

    report = run_drift_demo(
        n_pre=args.n_pre,
        n_post=args.n_post,
        shifts=args.shifts,
        seed=args.seed,
        report_path=args.out,
    )
    _print_report(report)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
