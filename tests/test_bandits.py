"""Tests for :mod:`nba.bandits` — the three pluggable exploration policies."""

from __future__ import annotations

import numpy as np
import pytest

from nba.bandits.base import Policy, sample_from_dist, softmax, validate_dist
from nba.bandits.epsilon_greedy import EpsilonGreedy
from nba.bandits.thompson import BootstrapEnsemble, ThompsonSampling
from nba.bandits.ucb import UCB, default_bucketizer
from nba.config import Settings
from nba.data.simulator import generate_logs, sample_context, sample_outcome, true_reward
from nba.reward.model import RewardModel
from nba.schema import ACTIONS, REWARD, Action, BanditEvent, ProspectContext

POLICY_NAMES = ["epsilon_greedy", "ucb", "thompson"]
_N = len(ACTIONS)


# --------------------------------------------------------------------------------------------- #
# Lightweight stand-ins so the contract tests don't need to fit any LightGBM models.
# --------------------------------------------------------------------------------------------- #
class _FakeModel:
    """Exposes ``q_all`` returning a fixed score vector (ignores context)."""

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)

    def q_all(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
        return self._scores.copy()


class _FakeEnsemble:
    """Exposes ``q_all_members`` returning a fixed ``(B, |A|)`` matrix."""

    def __init__(self, member_scores: np.ndarray) -> None:
        self._m = np.asarray(member_scores, dtype=np.float64)

    def q_all_members(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> np.ndarray:
        return self._m.copy()


def _ctx(seed: int = 1) -> ProspectContext:
    return sample_context(
        {"sale_price": 230_000.0, "year_built": 1995.0}, np.random.default_rng(seed)
    )


def _make_policy(name: str, rng: np.random.Generator) -> Policy:
    scores = np.array([0.5, 0.2, 0.1, 0.4, 0.3])
    if name == "epsilon_greedy":
        return EpsilonGreedy(_FakeModel(scores), epsilon=0.1, rng=rng)
    if name == "ucb":
        return UCB(_FakeModel(scores), c=1.0, temp=0.25, rng=rng)
    if name == "thompson":
        members = np.tile(scores, (8, 1)) + np.random.default_rng(0).normal(0, 0.05, (8, _N))
        return ThompsonSampling(_FakeEnsemble(members), rng=rng)
    raise ValueError(name)


# --------------------------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------------------------- #
def test_softmax_full_support_and_normalized() -> None:
    p = softmax(np.array([3.0, 1.0, -2.0]), temp=0.5)
    assert pytest.approx(p.sum()) == 1.0
    assert np.all(p > 0)
    assert int(np.argmax(p)) == 0


def test_softmax_rejects_nonpositive_temp() -> None:
    with pytest.raises(ValueError, match="temp"):
        softmax(np.array([1.0, 2.0]), temp=0.0)


def test_validate_dist_catches_bad_distributions() -> None:
    validate_dist({a: 1.0 / _N for a in ACTIONS})  # ok
    with pytest.raises(ValueError, match="sums to"):
        validate_dist({a: 0.1 for a in ACTIONS})
    with pytest.raises(ValueError, match="full support"):
        bad = {a: 0.0 for a in ACTIONS}
        bad[ACTIONS[0]] = 1.0
        validate_dist(bad)
    # a zero is allowed when full support is not required
    ok = {a: 0.0 for a in ACTIONS}
    ok[ACTIONS[0]] = 1.0
    validate_dist(ok, full_support=False)


def test_sample_from_dist_returns_listed_propensity() -> None:
    dist = {a: 0.1 for a in ACTIONS}
    dist[ACTIONS[0]] = 1.0 - 0.1 * (_N - 1)
    action, p = sample_from_dist(dist, np.random.default_rng(0))
    assert p == dist[action]


# --------------------------------------------------------------------------------------------- #
# Protocol + distribution contract (parametrized over all three policies)
# --------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", POLICY_NAMES)
def test_satisfies_policy_protocol(name: str) -> None:
    policy = _make_policy(name, np.random.default_rng(0))
    assert isinstance(policy, Policy)
    assert policy.name == name


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_action_dist_is_valid_full_support(name: str) -> None:
    policy = _make_policy(name, np.random.default_rng(0))
    dist = policy.action_dist(_ctx())
    assert set(dist) == set(ACTIONS)
    validate_dist(dist)  # sums to 1, all > 0
    assert pytest.approx(sum(dist.values())) == 1.0
    assert all(v > 0 for v in dist.values())


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_recommend_propensity_matches_dist(name: str) -> None:
    policy = _make_policy(name, np.random.default_rng(123))
    ctx = _ctx()
    # UCB mutates counts on recommend, so capture the decision-time dist first.
    dist = policy.action_dist(ctx)
    action, propensity = policy.recommend(ctx)
    assert propensity == pytest.approx(dist[action])


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_determinism_under_fixed_seed(name: str) -> None:
    def draw() -> list[Action]:
        policy = _make_policy(name, np.random.default_rng(7))
        return [policy.recommend(_ctx(i))[0] for i in range(25)]

    assert draw() == draw()


# --------------------------------------------------------------------------------------------- #
# ε-greedy specifics
# --------------------------------------------------------------------------------------------- #
def test_epsilon_greedy_limits() -> None:
    scores = np.array([0.1, 0.9, 0.2, 0.3, 0.4])  # arm 1 is best
    model = _FakeModel(scores)
    greedy = EpsilonGreedy(model, epsilon=0.0, rng=np.random.default_rng(0)).action_dist(_ctx())
    assert greedy[ACTIONS[1]] == pytest.approx(1.0)
    assert greedy[ACTIONS[0]] == pytest.approx(0.0)

    uniform = EpsilonGreedy(model, epsilon=1.0, rng=np.random.default_rng(0)).action_dist(_ctx())
    for a in ACTIONS:
        assert uniform[a] == pytest.approx(1.0 / _N)


def test_epsilon_greedy_splits_ties() -> None:
    scores = np.array([0.5, 0.5, 0.1, 0.1, 0.1])  # tie between arms 0 and 1
    dist = EpsilonGreedy(_FakeModel(scores), epsilon=0.2, rng=np.random.default_rng(0)).action_dist(
        _ctx()
    )
    explore = 0.2 / _N
    assert dist[ACTIONS[0]] == pytest.approx(explore + 0.8 / 2)
    assert dist[ACTIONS[1]] == pytest.approx(explore + 0.8 / 2)
    assert dist[ACTIONS[2]] == pytest.approx(explore)


def test_epsilon_greedy_rejects_bad_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedy(_FakeModel(np.zeros(_N)), epsilon=1.5, rng=np.random.default_rng(0))


# --------------------------------------------------------------------------------------------- #
# UCB specifics
# --------------------------------------------------------------------------------------------- #
def test_ucb_bonus_rewards_unpulled_then_shrinks() -> None:
    ucb = UCB(_FakeModel(np.zeros(_N)), c=1.0, temp=0.25, rng=np.random.default_rng(0))
    ctx = _ctx()
    bucket = default_bucketizer(ctx)
    pulled, unpulled = ACTIONS[0], ACTIONS[1]

    for _ in range(20):
        ucb.update(ctx, pulled)
    # An arm never pulled in this bucket has a larger optimism bonus than a heavily pulled one.
    assert ucb._bonus(bucket, unpulled) > ucb._bonus(bucket, pulled)

    before = ucb._bonus(bucket, unpulled)
    for _ in range(50):
        ucb.update(ctx, unpulled)
    assert ucb._bonus(bucket, unpulled) < before


def test_ucb_injected_bucketizer_is_used() -> None:
    ucb = UCB(
        _FakeModel(np.zeros(_N)),
        c=1.0,
        temp=0.25,
        rng=np.random.default_rng(0),
        bucketizer=lambda ctx: "one-bucket",
    )
    ucb.update(_ctx(1), ACTIONS[0])
    # Every context maps to the same bucket, so the count is visible from a different context.
    assert ucb._counts[("one-bucket", ACTIONS[0])] == 1


# --------------------------------------------------------------------------------------------- #
# Thompson specifics
# --------------------------------------------------------------------------------------------- #
def test_thompson_concentrates_on_dominant_arm_with_floor() -> None:
    members = np.tile(np.array([0.1, 0.1, 0.9, 0.1, 0.1]), (16, 1))  # arm 2 dominant for all
    ts = ThompsonSampling(_FakeEnsemble(members), rng=np.random.default_rng(0), floor=1e-3)
    dist = ts.action_dist(_ctx())
    assert dist[ACTIONS[2]] > 0.9
    assert all(v > 0 for v in dist.values())  # floor preserves overlap
    validate_dist(dist)


def test_thompson_rejects_negative_floor() -> None:
    with pytest.raises(ValueError, match="floor"):
        ThompsonSampling(_FakeEnsemble(np.zeros((4, _N))), rng=np.random.default_rng(0), floor=-1.0)


# --------------------------------------------------------------------------------------------- #
# Integration: bootstrap ensemble + "beats uniform" smoke test on a real reward model
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def trained() -> tuple[RewardModel, list[BanditEvent], Settings]:
    settings = Settings(seed=7)
    events = generate_logs(2500, settings=settings, seed=7)
    model = RewardModel.fit(events, settings=settings, val_frac=0.2)
    return model, events, settings


def test_bootstrap_ensemble_shape(trained: tuple[RewardModel, list[BanditEvent], Settings]) -> None:
    _, events, settings = trained
    ensemble = BootstrapEnsemble.fit(events, settings=settings, n_models=4)
    assert len(ensemble) == 4
    assert ensemble.q_all_members(_ctx()).shape == (4, _N)


def test_bootstrap_ensemble_save_load_roundtrip(
    trained: tuple[RewardModel, list[BanditEvent], Settings], tmp_path
) -> None:
    _, events, settings = trained
    ensemble = BootstrapEnsemble.fit(events, settings=settings, n_models=3)
    ensemble.save(tmp_path / "ens")
    reloaded = BootstrapEnsemble.load(tmp_path / "ens")
    assert np.allclose(ensemble.q_all_members(_ctx(5)), reloaded.q_all_members(_ctx(5)))


def test_policies_beat_uniform_baseline(
    trained: tuple[RewardModel, list[BanditEvent], Settings],
) -> None:
    model, events, settings = trained
    ensemble = BootstrapEnsemble.fit(events, settings=settings, n_models=4)

    ctx_rng = np.random.default_rng(321)
    contexts = [
        sample_context({"sale_price": float(p), "year_built": 1990.0}, ctx_rng)
        for p in ctx_rng.uniform(120_000.0, 320_000.0, size=400)
    ]

    # Exploration knobs are scaled to the calibrated reward range (rewards are O(0.1), so per-arm
    # q-gaps are ~0.1): the config defaults (c=1.0, temp=0.25) make UCB's softmax-of-bonuses far
    # too flat to exploit. A reward-scaled c/temp lets UCB tilt toward the better arms.
    policies: dict[str, Policy] = {
        "epsilon_greedy": EpsilonGreedy(model, epsilon=0.1, rng=np.random.default_rng(1)),
        "ucb": UCB(model, c=0.3, temp=0.1, rng=np.random.default_rng(2)),
        "thompson": ThompsonSampling(ensemble, rng=np.random.default_rng(3)),
    }

    def policy_value(policy: Policy) -> float:
        # Realized expected reward of the arms the policy actually plays (oracle-scored).
        return float(np.mean([true_reward(c, policy.recommend(c)[0]) for c in contexts]))

    uniform_rng = np.random.default_rng(99)
    uniform_value = float(
        np.mean([true_reward(c, ACTIONS[int(uniform_rng.integers(_N))]) for c in contexts])
    )

    for name, policy in policies.items():
        assert policy_value(policy) > uniform_value + 0.01, name


def test_online_loop_against_simulator(
    trained: tuple[RewardModel, list[BanditEvent], Settings],
) -> None:
    # A closed loop: recommend -> observe a sampled outcome -> (UCB) update counts. Compared
    # against uniform-random action selection over the *same* contexts with realized rewards.
    model, _, _ = trained
    ctx_rng = np.random.default_rng(11)
    contexts = [
        sample_context({"sale_price": 230_000.0, "year_built": 1990.0}, ctx_rng) for _ in range(300)
    ]

    policy = UCB(model, c=0.3, temp=0.1, rng=np.random.default_rng(0))
    policy_outcome_rng = np.random.default_rng(22)
    policy_rewards = []
    for ctx in contexts:
        action, propensity = policy.recommend(ctx)
        assert 0.0 < propensity <= 1.0
        policy_rewards.append(REWARD[sample_outcome(ctx, action, policy_outcome_rng)])

    uniform_action_rng = np.random.default_rng(0)
    uniform_outcome_rng = np.random.default_rng(22)
    uniform_rewards = []
    for ctx in contexts:
        action = ACTIONS[int(uniform_action_rng.integers(_N))]
        uniform_rewards.append(REWARD[sample_outcome(ctx, action, uniform_outcome_rng)])

    assert float(np.mean(policy_rewards)) > float(np.mean(uniform_rewards))
