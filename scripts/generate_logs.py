"""Generate logged bandit feedback from the D2D simulator and write it to parquet.

Usage:
    uv run python scripts/generate_logs.py --n 20000 --seed 7 --out data/logs.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba.config import get_settings
from nba.data.drift import generate_logs_for_settings
from nba.data.simulator import logs_to_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated logged bandit feedback.")
    parser.add_argument("--n", type=int, default=20_000, help="number of events to generate")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--out", type=Path, default=Path("data/logs.parquet"), help="output parquet path"
    )
    parser.add_argument(
        "--temp", type=float, default=0.5, help="logging-policy softmax temperature"
    )
    args = parser.parse_args()

    settings = get_settings()
    events, _ = generate_logs_for_settings(
        args.n, settings=settings, seed=args.seed, temp=args.temp
    )
    frame = logs_to_frame(events)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)

    arm_counts = frame["action"].value_counts().to_dict()
    print(f"wrote {len(frame)} events -> {args.out}")
    print(f"arm frequencies: {arm_counts}")
    print(f"mean reward:     {frame['reward'].mean():.4f}")
    print(f"min propensity:  {frame['propensity'].min():.4f}")


if __name__ == "__main__":
    main()
