"""Tests for :mod:`nba.data.simulator`."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nba.config import Settings
from nba.data.simulator import (
    action_distribution,
    behavior_policy,
    generate_logs,
    logs_to_frame,
    outcome_probs,
    sample_context,
    true_best_action,
    true_reward,
)
from nba.schema import ACTIONS, REWARD, Action, Outcome, ProspectContext

_REWARD_VALUES = set(REWARD.values())


def _hot_context() -> ProspectContext:
    """Evening, long-tenured, neighbor converted, wealthy — should favor engagement."""
    return ProspectContext(
        address_id="hot",
        lat=42.0,
        lon=-93.0,
        property_value=420_000.0,
        roof_age_years=15.0,
        est_income=160_000.0,
        tenure_years=20.0,
        prior_interactions=0,
        hour=18,
        dow=2,
        weather="clear",
        block_density=10.0,
        neighbor_recent_conversion=True,
        distance_from_rep_km=0.2,
        nearby_high_reward_density=0.9,
    )


def _cold_context() -> ProspectContext:
    """Mid-day, fatigued, bad weather — skipping should be competitive."""
    return ProspectContext(
        address_id="cold",
        lat=42.0,
        lon=-93.0,
        property_value=120_000.0,
        roof_age_years=2.0,
        est_income=28_000.0,
        tenure_years=0.5,
        prior_interactions=6,
        hour=12,
        dow=2,
        weather="rain",
        block_density=35.0,
        neighbor_recent_conversion=False,
        distance_from_rep_km=3.0,
        nearby_high_reward_density=0.1,
    )


def test_action_distribution_full_support_sums_to_one() -> None:
    dist = action_distribution(_hot_context())
    assert pytest.approx(sum(dist.values())) == 1.0
    assert all(p > 0 for p in dist.values())
    assert set(dist) == set(ACTIONS)


def test_behavior_policy_returns_matching_propensity() -> None:
    ctx = _hot_context()
    rng = np.random.default_rng(0)
    action, propensity = behavior_policy(ctx, rng)
    assert propensity == pytest.approx(action_distribution(ctx)[action])
    assert 0.0 < propensity <= 1.0


def test_outcome_probs_normalized() -> None:
    for action in ACTIONS:
        probs = outcome_probs(_hot_context(), action)
        assert pytest.approx(sum(probs.values())) == 1.0
        assert all(p >= 0 for p in probs.values())


def test_skip_door_is_near_zero_reward() -> None:
    assert true_reward(_hot_context(), Action.SKIP_DOOR) == pytest.approx(0.0, abs=1e-6)


def test_true_reward_within_reward_bounds() -> None:
    lo, hi = min(REWARD.values()), max(REWARD.values())
    for action in ACTIONS:
        assert lo <= true_reward(_hot_context(), action) <= hi


def test_oracle_rankings_make_sense() -> None:
    assert true_best_action(_hot_context()) is not Action.SKIP_DOOR
    cold_best = true_best_action(_cold_context())
    # In a hostile context, skipping should be at least as good as knocking.
    assert true_reward(_cold_context(), Action.SKIP_DOOR) >= true_reward(
        _cold_context(), Action.KNOCK_NOW
    )
    assert cold_best in ACTIONS


def test_generate_logs_reproducible(settings: Settings) -> None:
    a = logs_to_frame(generate_logs(200, settings=settings, seed=7))
    b = logs_to_frame(generate_logs(200, settings=settings, seed=7))
    import pandas as pd

    pd.testing.assert_frame_equal(a, b)


def test_generate_logs_positivity_and_coverage(settings: Settings) -> None:
    frame = logs_to_frame(generate_logs(2000, settings=settings, seed=7))
    assert (frame["propensity"] > 0).all()
    # every arm represented (overlap holds)
    logged_arms = set(frame["action"].unique())
    assert logged_arms == {a.value for a in ACTIONS}
    # rewards are drawn from the reward map
    assert set(frame["reward"].unique()).issubset(_REWARD_VALUES)
    # outcomes are valid
    assert set(frame["outcome"].unique()).issubset({o.value for o in Outcome})


def test_sample_context_produces_valid_context(settings: Settings) -> None:
    rng = np.random.default_rng(0)
    ctx = sample_context({"sale_price": 200_000.0, "year_built": 1990.0}, rng)
    assert isinstance(ctx, ProspectContext)
    assert 8 <= ctx.hour <= 20


def test_oracle_not_imported_by_downstream_modules() -> None:
    """Guard: reward/bandits/ope must not reference oracle symbols."""
    forbidden = {"latent_scores", "true_reward", "true_best_action", "outcome_probs"}
    src_root = Path(__file__).resolve().parents[1] / "src" / "nba"
    for sub in ("reward", "bandits", "ope"):
        pkg = src_root / sub
        if not pkg.exists():
            continue
        for py in pkg.rglob("*.py"):
            tree = ast.parse(py.read_text())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            leaked = forbidden & (names | attrs)
            assert not leaked, f"{py} references oracle symbols: {leaked}"
