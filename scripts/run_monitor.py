"""Score drift on the EventStore (or a parquet log) and append a DriftReport.

Offline batch tool — never touches the serve path. Honors ``NBA_USE_DRIFT_MONITORING``:
when disabled, exits cleanly with a message.

Usage:
    NBA_USE_DRIFT_MONITORING=1 uv run python scripts/run_monitor.py --db artifacts/events.db
    NBA_USE_DRIFT_MONITORING=1 uv run python scripts/run_monitor.py --logs data/logs.parquet
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
from nba.monitoring.retrain import bootstrap_deployed  # noqa: E402
from nba.monitoring.signals import DriftReportContext, build_drift_report  # noqa: E402
from nba.monitoring.store_reader import read_deployed_manifest  # noqa: E402
from nba.ope.estimators import LoggedBatch  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Score drift signals on logged events.")
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
        print(f"only {len(labeled)} labeled events; need at least 2 to score drift.")
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

    # Split reference/recent.
    recent_n = min(settings.monitor_recent_window, len(labeled) // 2)
    ref_n = min(settings.monitor_reference_window, max(1, len(labeled) - recent_n))
    reference = (
        labeled[-(recent_n + ref_n) : -recent_n]
        if len(labeled) > recent_n + ref_n
        else labeled[:-recent_n]
    )
    recent = labeled[-recent_n:]
    ref_batch = LoggedBatch.from_events(reference)
    recent_batch = LoggedBatch.from_events(recent)

    report = build_drift_report(
        ctx=DriftReportContext(
            model=model,
            policy=policy,
            reference=ref_batch,
            recent=recent_batch,
            deployed_dr=deployed_dr,
        ),
        settings=settings,
    )
    from nba.monitoring.signals import append_report  # noqa: PLC0415

    append_report(report, settings.monitoring_report_path)
    print(f"appended DriftReport to {settings.monitoring_report_path}")
    print(
        "  n_reference="
        f"{report.n_reference}  n_recent={report.n_recent}  overlap_ok={report.overlap_ok}"
    )
    for sig in report.signals:
        flag = "TRIGGERED" if sig.triggered else "ok"
        print(f"  {sig.name:<22} {sig.value:+.4f} (thr {sig.threshold:.3f}) [{flag}]")


if __name__ == "__main__":
    main()
