"""The ``Policy`` protocol and shared distribution helpers for bandit policies.

Every shipped policy implements two methods. ``recommend`` returns the ``(action, propensity)``
pair we *log* — the propensity is the probability the policy assigned to the chosen arm.
``action_dist`` returns the full distribution over actions that off-policy evaluation consumes.

**Why full support is mandatory.** IPS/DR divide by the *logging* propensity and reweight by the
*target* probability. A zero in the target is fine, but a zero in the logging policy means an
arm can be chosen that the estimator can never reweight — overlap is broken. To keep every
shipped policy safe to use as a logging policy too, all three floor their distributions so every
arm has strictly positive probability.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from nba.schema import ACTIONS, Action, ProspectContext


@runtime_checkable
class QModel(Protocol):
    """Anything that scores actions at a context (the reward model is the production one)."""

    def q_all(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
        """Return one score per action in ``actions`` order."""
        ...


@runtime_checkable
class QEnsemble(Protocol):
    """A posterior sample source: one score row per ensemble member."""

    def q_all_members(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> np.ndarray:
        """Return a ``(n_members, |actions|)`` matrix of per-member scores."""
        ...


@runtime_checkable
class Policy(Protocol):
    """A bandit policy: proposes an action to log and exposes its full action distribution."""

    name: str

    def recommend(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> tuple[Action, float]:
        """Return the chosen ``(action, propensity)`` for logging."""
        ...

    def action_dist(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> dict[Action, float]:
        """Return the policy's probability distribution over ``actions``."""
        ...


def validate_dist(
    dist: dict[Action, float], *, full_support: bool = True, tol: float = 1e-9
) -> None:
    """Raise ``ValueError`` unless ``dist`` sums to 1±tol, is non-negative, and (optionally) > 0."""
    values = np.array(list(dist.values()), dtype=np.float64)
    total = float(values.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(f"distribution sums to {total!r}, not 1")
    if np.any(values < -tol):
        raise ValueError("distribution has negative probabilities")
    if full_support and np.any(values <= 0.0):
        raise ValueError("distribution is not full support (an arm has probability 0)")


def sample_from_dist(dist: dict[Action, float], rng: np.random.Generator) -> tuple[Action, float]:
    """Draw one action from ``dist`` and return ``(action, dist[action])``."""
    actions = list(dist.keys())
    probs = np.array([dist[a] for a in actions], dtype=np.float64)
    idx = int(rng.choice(len(actions), p=probs))
    chosen = actions[idx]
    return chosen, float(dist[chosen])


def softmax(scores: np.ndarray, temp: float) -> np.ndarray:
    """Return a numerically stable, full-support softmax of ``scores`` at temperature ``temp``."""
    if temp <= 0.0:
        raise ValueError("temp must be > 0")
    z = np.asarray(scores, dtype=np.float64) / temp
    z -= z.max()
    exp = np.exp(z)
    return exp / exp.sum()
