"""The orchestrator: the seam where proposing meets disposing.

`recommend`/`feedback` drive the online bandit loop (and log every propensity); `plan_route`
turns a batch of doors into a walkable TSP-with-Profits plan. The profit fed to the router is the
bandit's *expected* value of a door -- the probability-weighted q over the policy's own action
distribution -- so exploration value flows into routing rather than a raw argmax. (Set
``argmax_profit=True`` to price doors by the greedy best action instead.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nba.api.store import EventStore
from nba.bandits.base import Policy, QModel
from nba.config import Settings
from nba.routing.distance import DistanceEngine
from nba.routing.tsp_profits import Route, solve_tsp_profits
from nba.schema import ACTIONS, Action, Outcome, ProspectContext


@dataclass(frozen=True)
class RecommendResult:
    """The outcome of a :meth:`Orchestrator.recommend` call."""

    decision_id: str
    action: Action
    propensity: float
    q_values: dict[Action, float]


class Orchestrator:
    """Wires a bandit policy, reward model, distance engine, and event store into one loop."""

    def __init__(
        self,
        *,
        policy: Policy,
        reward_model: QModel,
        distance_engine: DistanceEngine,
        store: EventStore,
        settings: Settings,
        argmax_profit: bool = False,
    ) -> None:
        self._policy = policy
        self._reward_model = reward_model
        self._distance_engine = distance_engine
        self._store = store
        self._settings = settings
        self._argmax_profit = argmax_profit

    @property
    def policy_name(self) -> str:
        """The active policy's name (surfaced on ``/health``)."""
        return self._policy.name

    def decision_count(self) -> int:
        """How many decisions have been logged."""
        return self._store.decision_count()

    def recommend(self, ctx: ProspectContext) -> RecommendResult:
        """Pick an action, log the decision (with its propensity), and return the result."""
        action, propensity = self._policy.recommend(ctx)
        decision_id = self._store.append_decision(
            context=ctx, action=action, propensity=propensity, policy_name=self._policy.name
        )
        q_row = self._reward_model.q_all(ctx)
        q_values = {a: float(v) for a, v in zip(ACTIONS, q_row, strict=True)}
        return RecommendResult(
            decision_id=decision_id,
            action=action,
            propensity=propensity,
            q_values=q_values,
        )

    def feedback(self, decision_id: str, outcome: Outcome) -> None:
        """Append the observed outcome for a prior decision (raises if the id is unknown)."""
        self._store.append_outcome(decision_id, outcome)

    def door_profit(self, ctx: ProspectContext) -> float:
        """Bandit-weighted value of a door: ``Σ_a action_dist(ctx)[a] · q(ctx, a)``.

        With ``argmax_profit=True`` this is instead the greedy best-action q.
        """
        q = self._reward_model.q_all(ctx)
        if self._argmax_profit:
            return float(q.max())
        dist = self._policy.action_dist(ctx)
        weights = np.array([dist[a] for a in ACTIONS], dtype=np.float64)
        return float((weights * q).sum())

    def plan_route(self, contexts: list[ProspectContext]) -> Route:
        """Plan a walkable route over ``contexts``, pricing each door by its bandit-weighted value.

        The depot is the centroid of the doors (a stand-in for the rep's current location);
        every door inherits the configured residential time window.
        """
        if not contexts:
            return Route(order=[], visited=[], dropped=[], total_time_s=0.0, total_profit=0.0)

        door_coords = [(c.lat, c.lon) for c in contexts]
        depot = (
            float(np.mean([lat for lat, _ in door_coords])),
            float(np.mean([lon for _, lon in door_coords])),
        )
        coords = [depot, *door_coords]
        profits = [0.0, *(self.door_profit(c) for c in contexts)]
        time_matrix = self._distance_engine.time_matrix(coords)

        open_s = self._settings.time_window[0] * 3600
        close_s = self._settings.time_window[1] * 3600
        time_windows = [(open_s, close_s)] * len(coords)

        return solve_tsp_profits(
            coords,
            profits,
            time_matrix,
            depot=0,
            capacity=self._settings.shift_capacity,
            time_windows=time_windows,
            drop_scale=self._settings.drop_scale,
            lambda_travel=self._settings.lambda_travel,
            seed=self._settings.seed,
        )

    def replan(self, remaining: list[ProspectContext]) -> Route:
        """Re-solve the route over the doors not yet visited."""
        return self.plan_route(remaining)
