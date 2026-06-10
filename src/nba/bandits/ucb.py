"""UCB: add an optimism bonus to ``q`` for rarely-pulled arms, then softmax for a smooth policy.

Classic UCB tracks per-arm pull counts and adds ``c·sqrt(ln t / n)`` so under-explored arms look
better than their point estimate. Contexts here are continuous and never repeat, so counts are
kept per **context bucket** (a coarse discretization of a few salient features) rather than per
raw context.

A hard ``argmax(q + bonus)`` would give a degenerate (one-hot) ``action_dist`` with no overlap,
so we instead **softmax** the UCB scores: a smooth, full-support distribution OPE can consume,
with ``temp`` controlling exploration sharpness.

The bucketizer is a pragmatic stand-in for LinUCB (which would model the bonus from a linear
context model directly); it is injectable so a richer scheme can drop in later.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable

import numpy as np

from nba.bandits.base import QModel, sample_from_dist, softmax
from nba.schema import ACTIONS, Action, ProspectContext


def default_bucketizer(ctx: ProspectContext) -> Hashable:
    """Coarse-bin a context into ``(time-of-day, property-value tier)`` for count bookkeeping."""
    if ctx.hour < 12:
        part = "morning"
    elif ctx.hour < 17:
        part = "afternoon"
    else:
        part = "evening"
    if ctx.property_value < 150_000.0:
        tier = "low"
    elif ctx.property_value < 250_000.0:
        tier = "mid"
    else:
        tier = "high"
    return part, tier


class UCB:
    """Softmax-of-UCB-scores policy with bucketed visit counts."""

    name = "ucb"

    def __init__(
        self,
        model: QModel,
        *,
        c: float,
        temp: float,
        rng: np.random.Generator,
        bucketizer: Callable[[ProspectContext], Hashable] | None = None,
    ) -> None:
        self._model = model
        self._c = float(c)
        self._temp = float(temp)
        self._rng = rng
        self._bucketizer = bucketizer or default_bucketizer
        self._counts: dict[tuple[Hashable, Action], int] = defaultdict(int)
        self._totals: dict[Hashable, int] = defaultdict(int)

    def _bonus(self, bucket: Hashable, action: Action) -> float:
        """Return the optimism bonus ``c·sqrt(ln(t+1) / (n+1))`` for ``action`` in ``bucket``."""
        t = self._totals[bucket]
        n = self._counts[(bucket, action)]
        return self._c * math.sqrt(math.log(t + 1) / (n + 1))

    def action_dist(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> dict[Action, float]:
        """Return ``softmax(q + bonus)`` over ``actions`` — smooth and full-support."""
        bucket = self._bucketizer(ctx)
        q = self._model.q_all(ctx, actions)
        bonuses = np.array([self._bonus(bucket, a) for a in actions], dtype=np.float64)
        probs = softmax(q + bonuses, self._temp)
        return dict(zip(actions, probs.tolist(), strict=True))

    def update(self, ctx: ProspectContext, action: Action) -> None:
        """Increment the visit count for ``action`` in ``ctx``'s bucket."""
        bucket = self._bucketizer(ctx)
        self._counts[(bucket, action)] += 1
        self._totals[bucket] += 1

    def recommend(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> tuple[Action, float]:
        """Sample from :meth:`action_dist`, bump the chosen arm's count, return ``(action, p)``."""
        dist = self.action_dist(ctx, actions)
        action, propensity = sample_from_dist(dist, self._rng)
        self.update(ctx, action)
        return action, propensity
