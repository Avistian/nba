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
    assert route.visited == []
    assert route.dropped == []
