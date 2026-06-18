"""A dataset-agnostic oracle facade used for grading (never for serving).

The flat and relational simulators expose the same conceptual oracle, but with slightly different
signatures: the relational oracle needs a :class:`~nba.data.relational_simulator.RelationalWorld`.
This module hides that difference behind a small :class:`Oracle` protocol so the demo and the
leaderboard can grade either dataset with identical call sites.

:class:`FlatOracle` is a pure pass-through to the flat simulator functions, so flat grading stays
numerically identical to calling them directly.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from nba.config import Settings
from nba.data import relational_simulator as rel
from nba.data import simulator as flat
from nba.data.relational_simulator import RelationalWorld
from nba.schema import Action, Outcome, ProspectContext


class Oracle(Protocol):
    """Ground-truth grading handle (eval-only)."""

    def true_reward(self, ctx: ProspectContext, action: Action) -> float: ...

    def true_best_action(self, ctx: ProspectContext) -> Action: ...

    def sample_outcome(
        self, ctx: ProspectContext, action: Action, rng: np.random.Generator
    ) -> Outcome: ...


class FlatOracle:
    """Pass-through to the flat simulator oracle (behaviour identical to calling it directly)."""

    def true_reward(self, ctx: ProspectContext, action: Action) -> float:
        return flat.true_reward(ctx, action)

    def true_best_action(self, ctx: ProspectContext) -> Action:
        return flat.true_best_action(ctx)

    def sample_outcome(
        self, ctx: ProspectContext, action: Action, rng: np.random.Generator
    ) -> Outcome:
        return flat.sample_outcome(ctx, action, rng)


class RelationalOracle:
    """Relational oracle bound to a fixed :class:`RelationalWorld`."""

    def __init__(self, world: RelationalWorld) -> None:
        self._world = world

    @property
    def world(self) -> RelationalWorld:
        return self._world

    def true_reward(self, ctx: ProspectContext, action: Action) -> float:
        return rel.true_reward(ctx, action, world=self._world)

    def true_best_action(self, ctx: ProspectContext) -> Action:
        return rel.true_best_action(ctx, world=self._world)

    def sample_outcome(
        self, ctx: ProspectContext, action: Action, rng: np.random.Generator
    ) -> Outcome:
        return rel.sample_outcome(ctx, action, rng, world=self._world)


def oracle_for(settings: Settings, *, world: RelationalWorld | None = None) -> Oracle:
    """Return the grading oracle for the active ``dataset_mode``.

    ``world`` is required in relational mode (the relational oracle is defined over a world).
    """
    if settings.dataset_mode == "relational":
        if world is None:
            raise ValueError("relational dataset_mode requires a RelationalWorld for grading")
        return RelationalOracle(world)
    return FlatOracle()
