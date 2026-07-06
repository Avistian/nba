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
from nba.bandits.base import Policy, QEnsemble, QModel
from nba.config import Settings
from nba.routing.distance import DistanceEngine, HaversineEngine, OSRMEngine
from nba.routing.tsp_profits import Route, solve_tsp_profits
from nba.schema import ACTIONS, Action, Outcome, ProspectContext


def build_distance_engine(settings: Settings) -> DistanceEngine:
    """Construct the travel-time engine selected by ``settings.distance_engine``.

    Defaults to :class:`HaversineEngine` (offline, no network); ``distance_engine="osrm"`` opts
    into the road-network :class:`OSRMEngine` at ``settings.osrm_url``.
    """
    if settings.distance_engine == "osrm":
        return OSRMEngine(settings.osrm_url)
    return HaversineEngine(speed_kmh=settings.walking_speed_kmh)


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
        reward_ensemble: QEnsemble | None = None,
    ) -> None:
        self._policy = policy
        self._reward_model = reward_model
        self._distance_engine = distance_engine
        self._store = store
        self._settings = settings
        self._argmax_profit = argmax_profit
        self._reward_ensemble = reward_ensemble
        if settings.use_risk_aware_routing and reward_ensemble is None:
            raise ValueError(
                "use_risk_aware_routing=True requires a reward_ensemble (the bootstrap ensemble "
                "supplies the per-door uncertainty); pass reward_ensemble=... to Orchestrator."
            )

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

    def door_profit_risk(self, ctx: ProspectContext) -> float:
        """Risk-adjusted door value: the mean price, discounted by the ensemble's uncertainty.

        Each ensemble member gives a full ``q`` row; the bandit-weighted value per member is a
        sample of the door's route-relevant value, and its spread *is* the model's uncertainty.
        Today's ``door_profit`` uses only the mean and throws that spread away; this method spends
        it:

        - ``mean_std`` (default): ``door_profit(ctx) - risk_kappa * std(v)`` -- a door the model
          loves on average but disagrees about wildly is worth less than an equally-valuable sure
          thing. The mean term is the deployed point estimate (:meth:`door_profit`), **not** the
          ensemble mean, so ``risk_kappa == 0.0`` reproduces ``door_profit`` *exactly* -- the flag
          is a true no-op until tuned.
        - ``cvar``: the mean of the worst ``cvar_alpha`` fraction of the per-member values -- a
          pragmatic per-door Conditional Value-at-Risk (an opt-in, strictly more conservative
          objective; it has no no-op setting). Full *route-value* CVaR over correlated scenarios is
          deferred to Phase 13's scenario machinery.
        """
        if self._reward_ensemble is None:
            raise ValueError("door_profit_risk requires a reward_ensemble")
        members = self._reward_ensemble.q_all_members(ctx)  # (B, |A|)
        dist = self._policy.action_dist(ctx)
        weights = np.array([dist[a] for a in ACTIONS], dtype=np.float64)
        per_member_value = members @ weights  # (B,)
        if self._settings.risk_objective == "cvar":
            alpha = self._settings.cvar_alpha
            k = max(1, int(np.ceil(alpha * per_member_value.size)))
            worst = np.sort(per_member_value)[:k]
            return float(worst.mean())
        std = float(per_member_value.std())
        return self.door_profit(ctx) - self._settings.risk_kappa * std

    def _door_price(self, ctx: ProspectContext) -> float:
        """Risk-adjusted door price when the flag is set, else the bandit-weighted mean."""
        if self._settings.use_risk_aware_routing:
            return self.door_profit_risk(ctx)
        return self.door_profit(ctx)

    def plan_route(self, contexts: list[ProspectContext]) -> Route | list[Route]:
        """Plan a walkable route over ``contexts``, pricing each door by its bandit-weighted value.

        With ``use_risk_aware_routing`` (Phase 11) the price is instead the ensemble's risk-adjusted
        value (:meth:`door_profit_risk`); ``risk_kappa == 0.0`` recovers the mean price exactly.

        The depot is the centroid of the doors (a stand-in for the rep's current location);
        every door inherits the configured residential time window.

        Returns a single :class:`Route` for one rep (``num_vehicles == 1``, the default), or one
        :class:`Route` per rep when ``num_vehicles > 1`` (Team Orienteering). With
        ``use_time_budget`` each rep's route is bounded by ``shift_hours``.
        """
        if not contexts:
            empty = Route(order=[], visited=[], dropped=[], total_time_s=0.0, total_profit=0.0)
            n_vehicles = self._settings.num_vehicles
            return [empty] * n_vehicles if n_vehicles > 1 else empty

        door_coords = [(c.lat, c.lon) for c in contexts]
        depot = (
            float(np.mean([lat for lat, _ in door_coords])),
            float(np.mean([lon for _, lon in door_coords])),
        )
        coords = [depot, *door_coords]
        profits = [0.0, *(self._door_price(c) for c in contexts)]
        time_matrix = self._distance_engine.time_matrix(coords)

        open_s = self._settings.time_window[0] * 3600
        close_s = self._settings.time_window[1] * 3600
        time_windows = [(open_s, close_s)] * len(coords)

        route_budget_s = (
            self._settings.shift_hours * 3600.0 if self._settings.use_time_budget else None
        )
        starts = list(self._settings.vehicle_starts) if self._settings.vehicle_starts else None
        ends = list(self._settings.vehicle_ends) if self._settings.vehicle_ends else None

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
            route_budget_s=route_budget_s,
            num_vehicles=self._settings.num_vehicles,
            starts=starts,
            ends=ends,
        )

    def replan(self, remaining: list[ProspectContext]) -> Route | list[Route]:
        """Re-solve the route over the doors not yet visited."""
        return self.plan_route(remaining)
