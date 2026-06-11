"""Off-policy-evaluate a candidate bandit policy against the logging baseline and gate promotion.

Usage:
    uv run python scripts/evaluate_policy.py --logs data/logs.parquet --model artifacts/models \
        --policy ucb --clip 100 --z 1.96 --min-lift 0.0

Exits non-zero if the gate rejects the candidate (so it can guard a CI promotion step).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nba.bandits.base import Policy
from nba.bandits.epsilon_greedy import EpsilonGreedy
from nba.bandits.thompson import BootstrapEnsemble, ThompsonSampling
from nba.bandits.ucb import UCB
from nba.config import get_settings
from nba.data.simulator import frame_to_events
from nba.ope.estimators import LoggedBatch, q_matrix
from nba.ope.gate import PromotionGate
from nba.reward.model import RewardModel

_POLICY_CHOICES = ("epsilon", "ucb", "thompson")


def _subsample(batch: LoggedBatch, max_rows: int, rng: np.random.Generator) -> LoggedBatch:
    """Return a random subset of the batch (≤ max_rows) for responsive scoring."""
    if len(batch) <= max_rows:
        return batch
    idx = rng.choice(len(batch), size=max_rows, replace=False)
    return LoggedBatch(
        contexts=[batch.contexts[i] for i in idx],
        actions=batch.actions[idx],
        rewards=batch.rewards[idx],
        propensities=batch.propensities[idx],
    )


def _build_policy(name: str, model: RewardModel, events: list, settings, rng) -> Policy:
    """Construct the named policy from config knobs (Thompson fits a bootstrap ensemble)."""
    if name == "epsilon":
        return EpsilonGreedy(model, epsilon=settings.epsilon, rng=rng)
    if name == "ucb":
        return UCB(model, c=settings.ucb_c, temp=settings.softmax_temp, rng=rng)
    if name == "thompson":
        ensemble = BootstrapEnsemble.fit(events, settings=settings, n_models=settings.n_bootstrap)
        return ThompsonSampling(ensemble, rng=rng)
    raise ValueError(f"unknown policy {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Off-policy evaluate a policy + gate promotion.")
    parser.add_argument("--logs", type=Path, default=Path("data/logs.parquet"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--policy", choices=_POLICY_CHOICES, default="ucb")
    parser.add_argument("--clip", type=float, default=None, help="importance-weight cap")
    parser.add_argument("--z", type=float, default=1.96, help="confidence multiplier")
    parser.add_argument("--min-lift", type=float, default=0.0, help="required lift over baseline")
    parser.add_argument(
        "--max-rows", type=int, default=10_000, help="subsample logs to at most this many rows"
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"seed": args.seed})
    rng = np.random.default_rng(args.seed)

    model = RewardModel.load(args.model)
    events = frame_to_events(pd.read_parquet(args.logs))
    full = LoggedBatch.from_events(events)
    baseline_value = float(full.rewards.mean())  # on-policy value of the logging policy
    batch = _subsample(full, args.max_rows, rng)

    policy = _build_policy(args.policy, model, events, settings, rng)
    q_hat = q_matrix(model, batch.contexts)

    gate = PromotionGate(z=args.z, min_lift=args.min_lift)
    decision = gate.evaluate(policy, batch, q_hat, baseline_value=baseline_value, clip=args.clip)

    print(f"policy           : {policy.name}")
    print(f"logged events    : {len(full):,} (scored on {len(batch):,})")
    print(f"baseline (logged): {baseline_value:.4f}")
    print(f"{'estimator':<8} {'value':>9} {'std_err':>9}   95%-CI")
    for name in ("ips", "snips", "dm", "dr"):
        res = decision.candidate[name]
        lo, hi = res.ci(args.z)
        print(f"{name:<8} {res.value:>9.4f} {res.std_err:>9.4f}   [{lo:.4f}, {hi:.4f}]")
    print()
    print(decision.reason)

    sys.exit(0 if decision.promote else 1)


if __name__ == "__main__":
    main()
