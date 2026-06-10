"""ε-greedy: exploit ``argmax q`` with probability ``1-ε``, explore uniformly with probability ε.

The simplest exploration baseline. The exploit mass ``1-ε`` goes to the best arm(s); the explore
mass ``ε`` is spread uniformly over *all* arms (so every arm keeps ``≥ ε/|A| > 0`` probability,
i.e. full support whenever ``ε > 0``).

**Tie rule:** if several arms share the maximum ``q`` (exact float equality), the exploit mass is
split *uniformly* among the tied arms. This is deterministic and needs no rng.
"""

from __future__ import annotations

import numpy as np

from nba.bandits.base import QModel, sample_from_dist
from nba.schema import ACTIONS, Action, ProspectContext


class EpsilonGreedy:
    """ε-greedy policy over any action scorer (a fitted reward model in production)."""

    name = "epsilon_greedy"

    def __init__(self, model: QModel, *, epsilon: float, rng: np.random.Generator) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self._model = model
        self._epsilon = float(epsilon)
        self._rng = rng

    def action_dist(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> dict[Action, float]:
        """Return ``ε/|A|`` per arm, plus the ``1-ε`` exploit mass split among argmax ties."""
        q = self._model.q_all(ctx, actions)
        n = len(actions)
        probs = np.full(n, self._epsilon / n, dtype=np.float64)
        best_mask = q == q.max()
        n_best = int(best_mask.sum())
        probs[best_mask] += (1.0 - self._epsilon) / n_best
        return dict(zip(actions, probs.tolist(), strict=True))

    def recommend(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> tuple[Action, float]:
        """Sample an action from :meth:`action_dist` and return ``(action, propensity)``."""
        return sample_from_dist(self.action_dist(ctx, actions), self._rng)
