"""Tests for :mod:`nba.ope` — estimators (vs OBP), variance ordering, and the promotion gate."""

from __future__ import annotations

import numpy as np
import pytest

from nba.config import Settings
from nba.data.simulator import generate_logs, sample_context, true_best_action
from nba.ope.estimators import (
    LoggedBatch,
    dr,
    evaluate_all,
    ips,
    q_matrix,
)
from nba.ope.gate import PromotionGate
from nba.reward.model import RewardModel
from nba.schema import ACTIONS, Action, ProspectContext

_N = len(ACTIONS)


def _ctx(seed: int = 1) -> ProspectContext:
    return sample_context(
        {"sale_price": 230_000.0, "year_built": 1995.0}, np.random.default_rng(seed)
    )


# --------------------------------------------------------------------------------------------- #
# Test policies (oracle one uses the simulator ground truth, but only to BUILD the target —
# never inside nba.ope itself).
# --------------------------------------------------------------------------------------------- #
class _FixedPolicy:
    """Always plays one fixed action (floored to full support)."""

    name = "fixed"

    def __init__(self, action: Action, floor: float = 1e-3) -> None:
        self._action = action
        self._floor = floor

    def action_dist(self, ctx, actions=ACTIONS):
        probs = {a: self._floor for a in actions}
        probs[self._action] = 1.0 - self._floor * (len(actions) - 1)
        return probs

    def recommend(self, ctx, actions=ACTIONS):
        return self._action, self.action_dist(ctx, actions)[self._action]


class _OraclePolicy:
    """Plays the oracle-optimal action (floored). Built from ground truth — test-only."""

    name = "oracle"

    def __init__(self, floor: float = 1e-3) -> None:
        self._floor = floor

    def action_dist(self, ctx, actions=ACTIONS):
        best = true_best_action(ctx)
        probs = {a: self._floor for a in actions}
        probs[best] = 1.0 - self._floor * (len(actions) - 1)
        return probs

    def recommend(self, ctx, actions=ACTIONS):
        best = true_best_action(ctx)
        return best, self.action_dist(ctx, actions)[best]


# --------------------------------------------------------------------------------------------- #
# Estimator correctness on controlled synthetic data
# --------------------------------------------------------------------------------------------- #
def test_ips_unbiased_when_target_equals_logging() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.1, 0.6, size=n)
    actions = rng.integers(0, _N, size=n)
    rewards = rng.normal(0.3, 0.2, size=n)
    # Target == logging: chosen-action prob equals the logging propensity, so every weight is 1.
    pi = np.full((n, _N), 0.0)
    for i in range(n):
        pi[i, :] = (1.0 - p[i]) / (_N - 1)
        pi[i, actions[i]] = p[i]
    batch = LoggedBatch(contexts=[_ctx()] * n, actions=actions, rewards=rewards, propensities=p)
    res = ips(batch, pi)
    assert res.value == pytest.approx(rewards.mean(), abs=1e-9)


def test_dr_variance_not_worse_than_ips() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    true_q = rng.uniform(0.0, 1.0, size=(n, _N))
    logging = rng.dirichlet(np.ones(_N), size=n)
    actions = np.array([rng.choice(_N, p=logging[i]) for i in range(n)])
    p = logging[np.arange(n), actions]
    rewards = true_q[np.arange(n), actions] + rng.normal(0.0, 0.1, size=n)
    target = rng.dirichlet(np.ones(_N) * 2.0, size=n)
    batch = LoggedBatch(contexts=[_ctx()] * n, actions=actions, rewards=rewards, propensities=p)
    # With a perfect q̂, DR's residual correction has far less variance than raw IPS.
    assert dr(batch, true_q, target).std_err <= ips(batch, target).std_err


# --------------------------------------------------------------------------------------------- #
# Numerical guards
# --------------------------------------------------------------------------------------------- #
def test_from_events_empty_raises() -> None:
    with pytest.raises(ValueError, match="no labeled events"):
        LoggedBatch.from_events([])


def test_zero_propensity_raises() -> None:
    with pytest.raises(ValueError, match="propensities"):
        LoggedBatch(
            contexts=[_ctx()],
            actions=np.array([0]),
            rewards=np.array([1.0]),
            propensities=np.array([0.0]),
        )


def test_pi_not_normalized_raises() -> None:
    batch = LoggedBatch(
        contexts=[_ctx()] * 3,
        actions=np.array([0, 1, 2]),
        rewards=np.array([0.1, 0.2, 0.3]),
        propensities=np.array([0.5, 0.5, 0.5]),
    )
    bad = np.full((3, _N), 0.1)  # rows sum to 0.5, not 1
    with pytest.raises(ValueError, match="sum to 1"):
        ips(batch, bad)


# --------------------------------------------------------------------------------------------- #
# Promotion gate (oracle-validated)
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def logged() -> tuple[LoggedBatch, np.ndarray, float]:
    settings = Settings(seed=7)
    events = generate_logs(3000, settings=settings, seed=7)
    model = RewardModel.fit(events, settings=settings, val_frac=0.2)
    batch = LoggedBatch.from_events(events)
    q_hat = q_matrix(model, batch.contexts)
    baseline = float(batch.rewards.mean())
    return batch, q_hat, baseline


def test_gate_rejects_worse_policy(logged: tuple[LoggedBatch, np.ndarray, float]) -> None:
    batch, q_hat, baseline = logged
    gate = PromotionGate(z=1.96, min_lift=0.0)
    decision = gate.evaluate(_FixedPolicy(Action.SKIP_DOOR), batch, q_hat, baseline_value=baseline)
    assert decision.promote is False
    assert decision.lift < 0.0


def test_gate_promotes_better_policy(logged: tuple[LoggedBatch, np.ndarray, float]) -> None:
    batch, q_hat, baseline = logged
    gate = PromotionGate(z=1.96, min_lift=0.0)
    decision = gate.evaluate(_OraclePolicy(), batch, q_hat, baseline_value=baseline)
    assert decision.promote is True
    assert decision.lower_bound > baseline


# --------------------------------------------------------------------------------------------- #
# Validation against the Open Bandit Pipeline (skipped unless obp is installed; see pyproject)
# --------------------------------------------------------------------------------------------- #
@pytest.mark.slow
def test_estimators_match_obp() -> None:
    pytest.importorskip("obp")
    from obp.dataset import SyntheticBanditDataset  # pyright: ignore[reportMissingImports]
    from obp.ope import (  # pyright: ignore[reportMissingImports]
        DirectMethod,
        DoublyRobust,
        InverseProbabilityWeighting,
        OffPolicyEvaluation,
    )

    dataset = SyntheticBanditDataset(
        n_actions=_N, dim_context=5, reward_type="continuous", random_state=7
    )
    fb = dataset.obtain_batch_bandit_feedback(n_rounds=3000)

    expected = fb["expected_reward"]  # (n, n_actions)
    z = expected - expected.max(axis=1, keepdims=True)
    pi = np.exp(z / 0.2)
    pi /= pi.sum(axis=1, keepdims=True)
    action_dist = pi[:, :, None]
    q = expected[:, :, None]

    ope = OffPolicyEvaluation(
        bandit_feedback=fb,
        ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust()],
    )
    obp_vals = ope.estimate_policy_values(action_dist=action_dist, estimated_rewards_by_reg_model=q)

    batch = LoggedBatch(
        contexts=[_ctx()] * fb["n_rounds"],
        actions=fb["action"].astype(np.int64),
        rewards=fb["reward"].astype(np.float64),
        propensities=fb["pscore"].astype(np.float64),
    )
    ours = evaluate_all(batch, pi, expected)

    for ours_key, obp_key in [("ips", "ipw"), ("dm", "dm"), ("dr", "dr")]:
        rel = abs(ours[ours_key].value - obp_vals[obp_key]) / (abs(obp_vals[obp_key]) + 1e-12)
        assert rel < 0.05, f"{ours_key}: ours={ours[ours_key].value} obp={obp_vals[obp_key]}"
