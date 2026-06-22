"""Full retrain loop: monitor → trigger → fit candidate → DR gate → promote/hold → audit.

Honors ``NBA_USE_DRIFT_MONITORING``: exits 0 with a message when disabled. When
enabled and no ``deployed.json`` exists, bootstraps a baseline from the labeled
log first.

Usage:
    NBA_USE_DRIFT_MONITORING=1 uv run python scripts/run_retrain_loop.py --db artifacts/events.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from nba.api.store import EventStore  # noqa: E402
from nba.bandits.epsilon_greedy import EpsilonGreedy  # noqa: E402
from nba.config import Settings  # noqa: E402
from nba.data.simulator import frame_to_events  # noqa: E402
from nba.monitoring.cadence import evaluate_monitor_cadence  # noqa: E402
from nba.monitoring.retrain import RetrainLoop, bootstrap_deployed  # noqa: E402
from nba.monitoring.store_reader import read_deployed_manifest  # noqa: E402
from nba.ope.gate import PromotionGate  # noqa: E402
from nba.reward.model import RewardModel  # noqa: E402
from nba.schema import BanditEvent  # noqa: E402


def _load_events(args: argparse.Namespace) -> list[BanditEvent]:
    if args.db:
        store = EventStore(Path(args.db))
        events = store.load_events()
        store.close()
        return events
    if args.logs:
        frame = pd.read_parquet(args.logs)
        return frame_to_events(frame)
    raise SystemExit("provide --db <path> or --logs <parquet>")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the drift monitor + conditional retrain loop."
    )
    parser.add_argument("--db", help="path to the EventStore SQLite DB")
    parser.add_argument("--logs", help="path to a parquet log frame")
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even when fewer than monitor_interval_events labeled rows arrived",
    )
    args = parser.parse_args()

    settings = Settings()
    if not settings.use_drift_monitoring:
        print("use_drift_monitoring=False; nothing to do (set NBA_USE_DRIFT_MONITORING=1).")
        return

    events = _load_events(args)
    labeled = [e for e in events if e.reward is not None]
    if len(labeled) < 2:
        print(f"only {len(labeled)} labeled events; need at least 2 to run the loop.")
        return

    cadence = evaluate_monitor_cadence(events, settings=settings)
    if not args.force and not cadence.due:
        print(
            "monitor cadence not met: "
            f"{cadence.events_since_last_report}/{cadence.interval} labeled events "
            "since last report; skipping (use --force to override)."
        )
        return

    # Load (or bootstrap) the deployed model + manifest.
    manifest = read_deployed_manifest(settings.deployed_model_manifest)
    if manifest is None or not Path(manifest.model_dir).exists():
        print("no deployed.json found; bootstrapping a baseline model...")
        model, policy, manifest, baseline_dr = bootstrap_deployed(settings=settings, events=labeled)
        deployed_dr = baseline_dr
    else:
        model = RewardModel.load(Path(manifest.model_dir))
        policy = EpsilonGreedy(
            model,
            epsilon=settings.epsilon,
            rng=__import__("numpy").random.default_rng(settings.seed),
        )
        deployed_dr = manifest.dr_value

    gate = PromotionGate(z=settings.ope_z, min_lift=settings.ope_min_lift)
    loop = RetrainLoop(settings=settings, gate=gate)
    outcome = loop.run(
        deployed_model=model, deployed_policy=policy, events=labeled, deployed_dr=deployed_dr
    )

    if outcome.report is None:
        print(outcome.gate_reason)
        return

    verdict = "PROMOTE" if outcome.promoted else "HOLD"
    print(f"retrain verdict: {verdict}")
    print(f"  trigger.should_retrain: {outcome.trigger.should_retrain}")
    print(f"  trigger.reasons: {outcome.trigger.reasons}")
    print(f"  trigger.overlap_ok: {outcome.trigger.overlap_ok}")
    if outcome.candidate_metrics:
        print(
            f"  candidate dr={outcome.candidate_metrics['dr']:+.4f} "
            f"(lb={outcome.candidate_metrics['dr_lb']:+.4f})"
        )
    print(f"  gate_reason: {outcome.gate_reason}")
    if outcome.candidate_model_dir:
        print(f"  candidate_model_dir: {outcome.candidate_model_dir}")


if __name__ == "__main__":
    main()
