"""Ethics guardrails: feature allow-list, sensitive-context exploration cap, no oracle leakage."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from nba.config import Settings
from nba.data.features import ALLOWED_FEATURES, FEATURE_NAMES, WEATHER_LEVELS, n_features
from nba.data.simulator import sample_context
from nba.ethics import EthicalPolicy, cap_exploration, is_sensitive
from nba.reward.model import RewardModel
from nba.schema import ACTIONS, Action, ProspectContext

# Fields that must never reach a model: protected/identity/geo.
_FORBIDDEN_FEATURES = {"lat", "lon", "address_id", "age", "race", "gender", "ethnicity", "zip"}
# Learning packages that must never see the simulator oracle.
_LEARNING_PACKAGES = ("reward", "bandits", "ope", "routing", "api", "pipeline")
_ORACLE_NAMES = {"true_reward", "latent_scores", "true_best_action", "outcome_probs"}
# Oracle-bearing modules (flat + relational) that learning code must never import from.
_ORACLE_MODULE_PREFIXES = ("nba.data.sim", "nba.data.relational_sim")


class _FakeModel:
    """Greedy model with a clear best arm so the modal mass is unambiguous."""

    def q_all(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
        return np.array([1.0, 0.2, 0.2, 0.2, 0.2])


def _ctx(prior_interactions: int) -> ProspectContext:
    base = sample_context({"sale_price": 230_000.0, "year_built": 1995.0}, np.random.default_rng(1))
    return base.model_copy(update={"prior_interactions": prior_interactions})


# --------------------------------------------------------------------------------------------- #
# Allow-list
# --------------------------------------------------------------------------------------------- #
def test_feature_allowlist_excludes_protected_and_geo() -> None:
    assert not (_FORBIDDEN_FEATURES & set(ALLOWED_FEATURES))
    assert not (_FORBIDDEN_FEATURES & set(FEATURE_NAMES))
    # The vector is exactly allow-list + weather one-hot + action one-hot — nothing else.
    assert n_features() == len(ALLOWED_FEATURES) + len(WEATHER_LEVELS) + len(ACTIONS)
    assert FEATURE_NAMES[: len(ALLOWED_FEATURES)] == list(ALLOWED_FEATURES)


def test_model_trains_only_on_allowed_features() -> None:
    from nba.data.simulator import generate_logs

    settings = Settings()
    events = generate_logs(400, settings=settings, seed=7)
    model = RewardModel.fit(events, settings=settings)
    # The persisted feature contract is the allow-list contract.
    assert model.feature_names == FEATURE_NAMES
    assert not (_FORBIDDEN_FEATURES & set(model.feature_names))


# --------------------------------------------------------------------------------------------- #
# Sensitive-context exploration cap
# --------------------------------------------------------------------------------------------- #
def test_cap_exploration_preserves_support_and_caps_mass() -> None:
    dist = {a: p for a, p in zip(ACTIONS, [0.6, 0.1, 0.1, 0.1, 0.1], strict=True)}
    capped = cap_exploration(dist, ceiling=0.05)
    assert abs(sum(capped.values()) - 1.0) < 1e-12
    assert all(p > 0.0 for p in capped.values())  # full support kept => OPE stays valid
    explore_mass = 1.0 - max(capped.values())
    assert explore_mass <= 0.05 + 1e-12


def test_ethical_policy_caps_exploration_in_sensitive_context() -> None:
    settings = Settings(
        cap_exploration_in_sensitive=True,
        sensitive_prior_interactions=4,
        sensitive_exploration_ceiling=0.05,
    )
    inner = _EpsilonInner(epsilon=0.4)
    policy = EthicalPolicy(inner, settings, rng=np.random.default_rng(0))

    sensitive = _ctx(prior_interactions=5)
    ordinary = _ctx(prior_interactions=0)
    assert is_sensitive(sensitive, settings)
    assert not is_sensitive(ordinary, settings)

    sens_dist = policy.action_dist(sensitive)
    explore_mass = 1.0 - max(sens_dist.values())
    assert explore_mass <= settings.sensitive_exploration_ceiling + 1e-9
    assert all(p > 0.0 for p in sens_dist.values())  # still full support

    # In an ordinary context the wrapper is a transparent pass-through.
    assert policy.action_dist(ordinary) == inner.action_dist(ordinary)


def test_ethical_policy_passthrough_when_flag_disabled() -> None:
    settings = Settings(cap_exploration_in_sensitive=False, sensitive_prior_interactions=4)
    inner = _EpsilonInner(epsilon=0.4)
    policy = EthicalPolicy(inner, settings, rng=np.random.default_rng(0))
    sensitive = _ctx(prior_interactions=9)
    assert policy.action_dist(sensitive) == inner.action_dist(sensitive)


class _EpsilonInner:
    """A minimal full-support, high-exploration policy with a single greedy arm."""

    name = "fake_eps"

    def __init__(self, epsilon: float) -> None:
        self._eps = epsilon
        self._model = _FakeModel()

    def action_dist(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> dict[Action, float]:
        q = self._model.q_all(ctx)
        best = int(np.argmax(q))
        n = len(actions)
        probs = np.full(n, self._eps / n)
        probs[best] += 1.0 - self._eps
        return dict(zip(actions, probs.tolist(), strict=True))

    def recommend(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> tuple[Action, float]:
        dist = self.action_dist(ctx, actions)
        best = max(dist, key=lambda a: dist[a])
        return best, dist[best]


# --------------------------------------------------------------------------------------------- #
# No oracle leakage (repo-wide AST scan)
# --------------------------------------------------------------------------------------------- #
def _iter_learning_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1] / "src" / "nba"
    files: list[Path] = []
    for pkg in _LEARNING_PACKAGES:
        files.extend((root / pkg).rglob("*.py"))
    assert files, "expected to find learning-package source files"
    return files


@pytest.mark.parametrize("path", _iter_learning_files(), ids=lambda p: p.name)
def test_no_oracle_leak(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            _ORACLE_MODULE_PREFIXES
        ):
            imported = {alias.name for alias in node.names}
            assert not (imported & _ORACLE_NAMES), f"{path.name} imports oracle: {imported}"
        if isinstance(node, ast.Name) and node.id in _ORACLE_NAMES:
            raise AssertionError(f"{path.name} references oracle symbol {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in _ORACLE_NAMES:
            raise AssertionError(f"{path.name} references oracle attribute {node.attr!r}")
