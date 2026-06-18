"""Run one named flag-config, score it on the leaderboard metrics, append a row, print the board.

Each run is graded against the baseline (all upgrade flags off = today's pipeline) and tagged a
**lift**, **regression**, or **neutral**. The leaderboard is append-only: corrections are new rows.

Usage:
    # Record the reference row once (all flags off).
    uv run python scripts/run_experiment.py --baseline-only

    # Grade an experiment (e.g. the relational dataset) against the baseline.
    uv run python scripts/run_experiment.py --experiment-id phase09-relational --phase 09 \\
        --dataset relational
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from nba.config import Settings
from nba.eval.leaderboard import (
    baseline_record,
    load_leaderboard,
    record_experiment,
    render_table,
)
from nba.eval.metrics import evaluate


def _apply_overrides(set_pairs: list[str], dataset: str | None) -> dict[str, object]:
    """Apply ``NBA_*=value`` pairs (and ``--dataset``) to the env; return the flag snapshot."""
    flags: dict[str, object] = {}
    if dataset is not None:
        os.environ["NBA_DATASET_MODE"] = dataset
        flags["NBA_DATASET_MODE"] = dataset
    for pair in set_pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects NBA_KEY=value, got {pair!r}")
        key, value = pair.split("=", 1)
        os.environ[key] = value
        flags[key] = value
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Run + grade one experiment on the leaderboard.")
    parser.add_argument("--experiment-id", default="baseline", help="human label for the run")
    parser.add_argument("--phase", default="baseline", help='"09".."16" or "baseline"')
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="NBA_KEY=value", help="config overrides"
    )
    parser.add_argument("--dataset", choices=["flat", "relational"], default=None)
    parser.add_argument("--baseline", default=None, help="experiment_id to compare against")
    parser.add_argument(
        "--baseline-only", action="store_true", help="record the all-flags-off reference row"
    )
    parser.add_argument("--n-shifts", type=int, default=None, help="override eval_n_shifts")
    parser.add_argument("--n-logs", type=int, default=4_000)
    parser.add_argument("--shift", type=int, default=24)
    parser.add_argument(
        "--md", type=Path, default=Path("artifacts/leaderboard.md"), help="markdown snapshot path"
    )
    args = parser.parse_args()

    if args.baseline_only:
        flags: dict[str, object] = {}
        experiment_id = "baseline"
        phase = "baseline"
    else:
        flags = _apply_overrides(args.set, args.dataset)
        experiment_id = args.experiment_id
        phase = args.phase

    settings = Settings()  # picks up the NBA_* overrides we just set
    n_shifts = args.n_shifts if args.n_shifts is not None else settings.eval_n_shifts

    print(
        f"evaluating '{experiment_id}' (phase {phase}, dataset={settings.dataset_mode}) "
        f"over {n_shifts} shift(s) x seeds {settings.eval_seeds} ..."
    )
    metrics = evaluate(
        settings=settings,
        n_shifts=n_shifts,
        seeds=settings.eval_seeds,
        n_logs=args.n_logs,
        shift=args.shift,
    )

    records = load_leaderboard(settings.leaderboard_path)
    if args.baseline_only:
        baseline = None
    else:
        baseline = baseline_record(records, args.baseline or settings.baseline_experiment_id)
        if baseline is None:
            print(
                "warning: no baseline row found; recording this run as its own reference "
                "(run with --baseline-only first for a real comparison)."
            )

    record = record_experiment(
        metrics,
        settings=settings,
        experiment_id=experiment_id,
        phase=phase,
        flags=flags,
        baseline=baseline,
    )

    board = load_leaderboard(settings.leaderboard_path)
    table = render_table(board)
    print(
        f"\nrecorded '{record.experiment_id}': verdict = {record.verdict.upper()} "
        f"(gate {'PASS' if record.gate_passed else 'FAIL'})\n"
    )
    print(table)

    if args.md is not None:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(f"# Experiment leaderboard\n\n{table}\n", encoding="utf-8")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
