"""Live-streaming drift demo: events tick in over wall-clock time → monitor fires → alert → retrain.

Producer-driven narrative (isolated ``artifacts/drift_demo/*`` tree):

1. **Warm-up** — generate in-distribution logs, bootstrap deployed model.
2. **Stream** — each tick ingests labeled events (drifting after ``--drift-onset``).
3. **Monitor** — ``RetrainLoop.run`` once per tick; one ``DriftReport`` per tick.
4. **Alert** — ``maybe_alert_drift`` emails (or dry-run prints) on significant drift.
5. **Recover** — post-promote ticks use monitor-only scoring.

CLI:
    NBA_USE_DRIFT_MONITORING=1 uv run python scripts/run_online_drift_demo.py \\
        --warmup 3000 --events-per-tick 200 --ticks 12 --tick-seconds 15
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from drift_demo_common import (  # noqa: E402
    ingest_events_incremental,
    load_promoted_stack,
    online_demo_settings,
    reset_demo_tree,
    score_drift,
    signals_from_report,
)

from nba.config import Settings  # noqa: E402
from nba.data.drift import DriftSpec, generate_logs_with_drift  # noqa: E402
from nba.data.simulator import generate_logs  # noqa: E402
from nba.monitoring.alerting import AlertDecision, maybe_alert_drift  # noqa: E402
from nba.monitoring.retrain import RetrainLoop, bootstrap_deployed  # noqa: E402
from nba.monitoring.store_reader import read_deployed_manifest  # noqa: E402
from nba.ope.gate import PromotionGate  # noqa: E402
from nba.schema import BanditEvent  # noqa: E402

# Canonical post-drift world (matches batch demo).
_MAX_DRIFT = DriftSpec(at_fraction=0.0, reward_scale=1.3, knock_evening_boost=0.15)


@dataclass(frozen=True)
class TickOutcome:
    """One live tick's monitor + alert + promote verdict."""

    tick: int
    n_new_events: int
    n_total_labeled: int
    promoted: bool
    trigger_fired: bool
    trigger_reasons: tuple[str, ...]
    reward_psi: float
    alert: AlertDecision
    phase: str  # "warmup" | "pre_drift" | "drifting" | "post_retrain"


def drift_spec_for_tick(
    tick: int,
    *,
    mode: str,
    onset: int,
    ticks: int,
    max_spec: DriftSpec = _MAX_DRIFT,
) -> DriftSpec | None:
    """Return the drift spec for tick ``tick``, or ``None`` before onset."""
    if tick < onset:
        return None
    if mode == "step":
        return max_spec
    # ramp: scale drift strength from 0 → 1 over remaining ticks
    remaining = max(1, ticks - onset)
    scale = min(1.0, (tick - onset + 1) / remaining)
    return DriftSpec(
        at_fraction=0.0,
        reward_scale=1.0 + (max_spec.reward_scale - 1.0) * scale,
        knock_evening_boost=max_spec.knock_evening_boost * scale,
        weather_slam_mult=1.0 + (max_spec.weather_slam_mult - 1.0) * scale,
    )


def restamp_events(events: list[BanditEvent], clock: datetime) -> list[BanditEvent]:
    """Re-stamp events to wall-clock time so monitor windows and cadence work live."""
    stamped: list[BanditEvent] = []
    for i, event in enumerate(events):
        ts = clock + timedelta(seconds=i)
        stamped.append(
            event.model_copy(
                update={
                    "timestamp": ts,
                    "decision_id": str(uuid.uuid4()),
                }
            )
        )
    return stamped


def generate_tick_events(
    n: int,
    *,
    settings: Settings,
    seed: int,
    spec: DriftSpec | None,
) -> list[BanditEvent]:
    """Generate one tick's labeled events (stationary or drifting)."""
    if spec is None:
        return generate_logs(n, settings=settings, seed=seed)
    return generate_logs_with_drift(n, settings=settings, seed=seed, spec=spec)


def run_one_tick(
    *,
    tick: int,
    settings: Settings,
    cumulative: list[BanditEvent],
    deployed_model,
    deployed_policy,
    baseline_dr: float,
    loop: RetrainLoop,
    promoted: bool,
    events_per_tick: int,
    drift_mode: str,
    drift_onset: int,
    total_ticks: int,
    seed: int,
    send_email: bool,
    clock: datetime | None = None,
) -> tuple[TickOutcome, list[BanditEvent], object, object, float, bool]:
    """Ingest one tick, run monitor/retrain, optionally alert; return updated state."""
    now = clock or datetime.now(UTC)
    spec = drift_spec_for_tick(
        tick, mode=drift_mode, onset=drift_onset, ticks=total_ticks
    )
    raw = generate_tick_events(
        events_per_tick,
        settings=settings,
        seed=seed + tick * 10_000,
        spec=spec,
    )
    stamped = restamp_events(raw, now)
    n_ingested = ingest_events_incremental(settings, stamped)
    cumulative = list(cumulative) + stamped

    if promoted:
        manifest = read_deployed_manifest(settings.deployed_model_manifest)
        promoted_at = manifest.promoted_at if manifest else None
        signals, _overlap = score_drift(
            model=deployed_model,
            policy=deployed_policy,
            events=cumulative,
            settings=settings,
            deployed_dr=baseline_dr,
            promoted_at=promoted_at,
        )
        phase = "post_retrain"
        outcome = None
        trigger_fired = False
        trigger_reasons: tuple[str, ...] = ()
        tick_promoted = False
        drift_report = None
        trigger = None
    else:
        outcome = loop.run(
            deployed_model=deployed_model,
            deployed_policy=deployed_policy,
            events=cumulative,
            deployed_dr=baseline_dr,
        )
        drift_report = outcome.report
        trigger = outcome.trigger
        trigger_fired = trigger.should_retrain
        trigger_reasons = trigger.reasons
        tick_promoted = outcome.promoted
        if drift_report is not None:
            signals, _overlap = signals_from_report(drift_report)
        else:
            signals = {}
        phase = "drifting" if tick >= drift_onset else "pre_drift"

        if tick_promoted and outcome.candidate_model_dir is not None:
            deployed_model, deployed_policy, baseline_dr = load_promoted_stack(
                settings=settings,
                events=cumulative,
            )
            promoted = True
            phase = "post_retrain"

    alert = AlertDecision(sent=False, reason="skipped")
    if send_email and drift_report is not None and trigger is not None:
        alert = maybe_alert_drift(
            drift_report,
            trigger,
            outcome,
            settings=settings,
            now=now,
        )
    elif not send_email:
        alert = AlertDecision(sent=False, reason="no_email_flag")

    tick_outcome = TickOutcome(
        tick=tick,
        n_new_events=n_ingested,
        n_total_labeled=len([e for e in cumulative if e.reward is not None]),
        promoted=tick_promoted,
        trigger_fired=trigger_fired,
        trigger_reasons=trigger_reasons,
        reward_psi=signals.get("reward_psi", 0.0),
        alert=alert,
        phase=phase,
    )
    return (
        tick_outcome,
        cumulative,
        deployed_model,
        deployed_policy,
        baseline_dr,
        promoted,
    )


def run_online_drift_demo(
    *,
    warmup: int = 3000,
    events_per_tick: int = 200,
    ticks: int = 12,
    tick_seconds: float = 15.0,
    drift_mode: str = "ramp",
    drift_onset: int = 4,
    seed: int = 7,
    reset: bool = True,
    send_email: bool = True,
    settings: Settings | None = None,
) -> list[TickOutcome]:
    """Run the full live-streaming drift demo."""
    base = (settings or Settings()).model_copy(
        update={"seed": seed, "use_drift_monitoring": True}
    )
    settings = online_demo_settings(base)
    settings.ensure_dirs()

    if reset:
        reset_demo_tree(settings)

    np.random.default_rng(seed)
    now = datetime.now(UTC)
    warmup_raw = generate_logs(warmup, settings=settings, seed=seed)
    warmup_events = restamp_events(warmup_raw, now - timedelta(seconds=warmup))
    ingest_events_incremental(settings, warmup_events, policy_name="online_drift_warmup")

    deployed_model, deployed_policy, _manifest, baseline_dr = bootstrap_deployed(
        settings=settings, events=warmup_events
    )
    gate = PromotionGate(z=settings.ope_z, min_lift=settings.ope_min_lift)
    loop = RetrainLoop(settings=settings, gate=gate)

    cumulative: list[BanditEvent] = list(warmup_events)
    promoted = False
    records: list[TickOutcome] = []

    print(
        f"online drift demo  seed={seed}  warmup={warmup:,}  "
        f"events/tick={events_per_tick}  ticks={ticks}  "
        f"drift={drift_mode}@tick{drift_onset}"
    )
    print(f"artifacts -> {settings.db_path.parent}")
    print(f"baseline DR: {baseline_dr:+.4f}\n")

    for tick in range(ticks):
        tick_outcome, cumulative, deployed_model, deployed_policy, baseline_dr, promoted = (
            run_one_tick(
                tick=tick,
                settings=settings,
                cumulative=cumulative,
                deployed_model=deployed_model,
                deployed_policy=deployed_policy,
                baseline_dr=baseline_dr,
                loop=loop,
                promoted=promoted,
                events_per_tick=events_per_tick,
                drift_mode=drift_mode,
                drift_onset=drift_onset,
                total_ticks=ticks,
                seed=seed,
                send_email=send_email,
            )
        )
        records.append(tick_outcome)
        _print_tick(tick_outcome)
        if tick < ticks - 1 and tick_seconds > 0:
            time.sleep(tick_seconds)

    return records


def _print_tick(outcome: TickOutcome) -> None:
    reasons = ",".join(outcome.trigger_reasons) if outcome.trigger_reasons else "-"
    verdict = "PROMOTE" if outcome.promoted else "hold"
    print(
        f"tick {outcome.tick:>2}  {outcome.phase:<12}  "
        f"+{outcome.n_new_events} events  total={outcome.n_total_labeled:,}  "
        f"psi={outcome.reward_psi:+.4f}  trigger={outcome.trigger_fired} ({reasons})  "
        f"{verdict}  alert={outcome.alert.reason}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-streaming drift monitor demo.")
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--events-per-tick", type=int, default=200)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--tick-seconds", type=float, default=15.0)
    parser.add_argument("--drift-mode", choices=("ramp", "step"), default="ramp")
    parser.add_argument("--drift-onset", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-email", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    if not settings.use_drift_monitoring:
        print("use_drift_monitoring=False; nothing to do (set NBA_USE_DRIFT_MONITORING=1).")
        return

    run_online_drift_demo(
        warmup=args.warmup,
        events_per_tick=args.events_per_tick,
        ticks=args.ticks,
        tick_seconds=args.tick_seconds,
        drift_mode=args.drift_mode,
        drift_onset=args.drift_onset,
        seed=args.seed,
        reset=args.reset,
        send_email=not args.no_email,
        settings=settings.model_copy(update={"use_drift_monitoring": True}),
    )


if __name__ == "__main__":
    main()
