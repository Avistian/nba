"""Tests for the :class:`Orchestrator` -- the bandit-profits-to-router seam."""

from __future__ import annotations

import numpy as np
import pytest

from nba.api.store import EventStore
from nba.bandits.epsilon_greedy import EpsilonGreedy
from nba.config import Settings
from nba.data.simulator import sample_context
from nba.pipeline.orchestrator import Orchestrator
from nba.routing.distance import HaversineEngine
from nba.schema import ACTIONS, Action, Outcome, ProspectContext

_SCORES = np.array([1.00, 0.90, 0.85, 0.95, 0.88])


class _FakeModel:
    """Context-independent ``q_all`` so door profits are predictable."""

    def q_all(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
        return _SCORES.copy()


def _base_ctx() -> ProspectContext:
    return sample_context({"sale_price": 230_000.0, "year_built": 1995.0}, np.random.default_rng(1))


def _door(base: ProspectContext, address_id: str, lat: float, lon: float) -> ProspectContext:
    return base.model_copy(update={"address_id": address_id, "lat": lat, "lon": lon})


def _orchestrator(settings: Settings) -> Orchestrator:
    rng = np.random.default_rng(0)
    return Orchestrator(
        policy=EpsilonGreedy(_FakeModel(), epsilon=settings.epsilon, rng=rng),
        reward_model=_FakeModel(),
        distance_engine=HaversineEngine(speed_kmh=settings.walking_speed_kmh),
        store=EventStore(settings.db_path),
        settings=settings,
    )


def test_recommend_logs_one_decision(settings: Settings) -> None:
    orch = _orchestrator(settings)
    result = orch.recommend(_base_ctx())

    assert result.decision_id
    assert 0.0 < result.propensity <= 1.0
    assert set(result.q_values) == set(ACTIONS)
    assert orch.decision_count() == 1


def test_door_profit_is_bandit_weighted_q(settings: Settings) -> None:
    orch = _orchestrator(settings)
    ctx = _base_ctx()

    dist = orch._policy.action_dist(ctx)  # noqa: SLF001 - white-box check of the weighting
    expected = sum(dist[a] * _SCORES[i] for i, a in enumerate(ACTIONS))

    assert orch.door_profit(ctx) == pytest.approx(expected)
    # weighted value must sit between the worst and best arm.
    assert _SCORES.min() <= orch.door_profit(ctx) <= _SCORES.max()


def test_plan_route_keeps_cluster_drops_far_door(settings: Settings) -> None:
    orch = _orchestrator(settings)
    base = _base_ctx()
    # A tight cluster of doors plus one far outlier (same profit for all).
    cluster = [_door(base, f"near-{k}", 42.000 + 0.0005 * k, -93.600) for k in range(5)]
    far = _door(base, "far", 42.030, -93.600)  # a few km north of the cluster
    contexts = [*cluster, far]

    route = orch.plan_route(contexts)
    assert not isinstance(route, list)

    # node indices: 0 = depot, then contexts in order; the far door is the last index.
    far_index = len(contexts)
    assert far_index in route.dropped
    assert len(route.visited) >= 4


def test_feedback_produces_labeled_event(settings: Settings) -> None:
    orch = _orchestrator(settings)
    result = orch.recommend(_base_ctx())

    orch.feedback(result.decision_id, Outcome.APPOINTMENT)

    events = orch._store.load_events()  # noqa: SLF001 - verifying the persisted label
    (event,) = events
    assert event.outcome is Outcome.APPOINTMENT
    assert event.reward is not None


def test_empty_route_is_safe(settings: Settings) -> None:
    orch = _orchestrator(settings)
    route = orch.plan_route([])
    assert not isinstance(route, list)
    assert route.visited == []
    assert route.dropped == []


# --- Phase 11: risk-aware routing -------------------------------------------------------------


class _EnsembleModel:
    """A reward model + bootstrap ensemble in one, keyed by ``address_id``.

    ``q_all`` returns the per-action mean over members (so it matches the ensemble mean exactly,
    making ``risk_kappa == 0`` a clean no-op). Every member row is constant across actions, so the
    policy-weighted per-member value equals that member's scalar -- letting the tests control each
    door's uncertainty directly.
    """

    def __init__(self, members_by_addr: dict[str, np.ndarray]) -> None:
        self._by_addr = members_by_addr

    def q_all(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
        return self._by_addr[ctx.address_id].mean(axis=0)

    def q_all_members(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> np.ndarray:
        return self._by_addr[ctx.address_id].copy()


def _const_members(value: float, n_members: int = 8) -> np.ndarray:
    """A zero-spread ensemble: every member scores ``value`` on every action."""
    return np.full((n_members, len(ACTIONS)), value, dtype=np.float64)


def _spread_members(hi: float, lo: float, n_members: int = 8) -> np.ndarray:
    """A high-spread ensemble: half the members score ``hi``, half ``lo`` (mean = midpoint)."""
    half = n_members // 2
    return np.vstack(
        [
            np.full((half, len(ACTIONS)), hi, dtype=np.float64),
            np.full((n_members - half, len(ACTIONS)), lo, dtype=np.float64),
        ]
    )


def _risk_orchestrator(settings: Settings, model: _EnsembleModel) -> Orchestrator:
    rng = np.random.default_rng(0)
    return Orchestrator(
        policy=EpsilonGreedy(model, epsilon=settings.epsilon, rng=rng),
        reward_model=model,
        distance_engine=HaversineEngine(speed_kmh=settings.walking_speed_kmh),
        store=EventStore(settings.db_path),
        settings=settings,
        reward_ensemble=model,
    )


def test_risk_routing_requires_ensemble(settings: Settings) -> None:
    s = settings.model_copy(update={"use_risk_aware_routing": True})
    with pytest.raises(ValueError, match="reward_ensemble"):
        Orchestrator(
            policy=EpsilonGreedy(_FakeModel(), epsilon=s.epsilon, rng=np.random.default_rng(0)),
            reward_model=_FakeModel(),
            distance_engine=HaversineEngine(speed_kmh=s.walking_speed_kmh),
            store=EventStore(s.db_path),
            settings=s,
        )


def test_door_profit_risk_recovers_mean_at_kappa_zero(settings: Settings) -> None:
    s = settings.model_copy(update={"use_risk_aware_routing": True, "risk_kappa": 0.0})
    model = _EnsembleModel({"d": _spread_members(0.9, 0.1)})  # spread present, but kappa=0
    orch = _risk_orchestrator(s, model)
    ctx = _door(_base_ctx(), "d", 42.0, -93.6)

    assert orch.door_profit_risk(ctx) == pytest.approx(orch.door_profit(ctx))


def test_door_profit_risk_discounts_uncertainty(settings: Settings) -> None:
    s = settings.model_copy(update={"use_risk_aware_routing": True, "risk_kappa": 1.0})
    model = _EnsembleModel(
        {
            "safe": _const_members(0.5),  # mean 0.5, std 0.0
            "risky": _spread_members(0.9, 0.1),  # mean 0.5, std 0.4 (same mean, more spread)
        }
    )
    orch = _risk_orchestrator(s, model)
    safe = _door(_base_ctx(), "safe", 42.0, -93.6)
    risky = _door(_base_ctx(), "risky", 42.0, -93.6)

    # Identical mean value, but the uncertain door is priced strictly lower.
    assert orch.door_profit(safe) == pytest.approx(orch.door_profit(risky))
    assert orch.door_profit_risk(safe) == pytest.approx(0.5)
    assert orch.door_profit_risk(risky) == pytest.approx(0.1)
    assert orch.door_profit_risk(risky) < orch.door_profit_risk(safe)


def test_door_profit_risk_cvar_uses_worst_tail(settings: Settings) -> None:
    s = settings.model_copy(
        update={"use_risk_aware_routing": True, "risk_objective": "cvar", "cvar_alpha": 0.25}
    )
    model = _EnsembleModel({"d": _spread_members(0.9, 0.1, n_members=8)})
    orch = _risk_orchestrator(s, model)
    ctx = _door(_base_ctx(), "d", 42.0, -93.6)

    # Worst 25% of 8 members = the 2 lowest (both 0.1) => CVaR 0.1, well below the 0.5 mean.
    assert orch.door_profit_risk(ctx) == pytest.approx(0.1)
    assert orch.door_profit_risk(ctx) < orch.door_profit(ctx)


def test_risk_router_drops_uncertain_door_first(settings: Settings) -> None:
    # Capacity forces exactly one of two equidistant doors; risk pricing keeps the sure one.
    s = settings.model_copy(
        update={"use_risk_aware_routing": True, "risk_kappa": 1.0, "shift_capacity": 1}
    )
    model = _EnsembleModel({"safe": _const_members(0.5), "risky": _spread_members(0.9, 0.1)})
    orch = _risk_orchestrator(s, model)
    base = _base_ctx()
    safe = _door(base, "safe", 42.0000, -93.6000)
    risky = _door(base, "risky", 42.0002, -93.6000)

    route = orch.plan_route([safe, risky])
    assert not isinstance(route, list)

    # coords = [depot, safe, risky]; node 1 is the sure door, node 2 the uncertain one.
    assert 1 in route.visited
    assert 2 in route.dropped


def test_risk_aware_route_reduces_realized_value_variance(settings: Settings) -> None:
    # Six clustered doors, capacity 3. The uncertain doors carry a hair more mean value, so mean
    # pricing prefers them; risk pricing prefers the sure doors. Scoring each ensemble member as a
    # value scenario, the risk-aware selection has far lower cross-scenario variance at a
    # comparable mean -- the doc 11 §10 robustness claim.
    model = _EnsembleModel(
        {
            **{f"safe-{k}": _const_members(0.50) for k in range(3)},
            **{f"risky-{k}": _spread_members(0.91, 0.11) for k in range(3)},  # mean .51, std .40
        }
    )
    base = _base_ctx()
    doors = [_door(base, f"safe-{k}", 42.0 + 1e-4 * k, -93.6) for k in range(3)]
    doors += [_door(base, f"risky-{k}", 42.0 + 1e-4 * (k + 3), -93.6) for k in range(3)]

    def visited_for(kappa: float) -> list[int]:
        s = settings.model_copy(
            update={
                "use_risk_aware_routing": kappa > 0.0,  # kappa 0 => plain mean pricing baseline
                "risk_kappa": kappa,
                "shift_capacity": 3,
            }
        )
        route = _risk_orchestrator(s, model).plan_route(doors)
        assert not isinstance(route, list)
        return route.visited

    def realized_scenarios(visited: list[int], n_members: int = 8) -> np.ndarray:
        # Each ensemble member is a scenario of the doors' true value; row is constant across
        # actions so the member's value is just its (scalar) row mean.
        totals = [
            sum(float(model.q_all_members(doors[node - 1])[m].mean()) for node in visited)
            for m in range(n_members)
        ]
        return np.asarray(totals, dtype=np.float64)

    mean_totals = realized_scenarios(visited_for(0.0))
    risk_totals = realized_scenarios(visited_for(1.0))

    assert risk_totals.std() < mean_totals.std()  # the robustness win
    assert risk_totals.mean() >= 0.9 * mean_totals.mean()  # at comparable mean
