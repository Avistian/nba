"""Tests for the FastAPI service via ``TestClient`` on an injected orchestrator."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nba.api.app import build_app
from nba.api.store import EventStore
from nba.bandits.epsilon_greedy import EpsilonGreedy
from nba.config import Settings
from nba.data.simulator import sample_context
from nba.pipeline.orchestrator import Orchestrator
from nba.routing.distance import HaversineEngine
from nba.schema import ACTIONS, Action, ProspectContext


class _FakeModel:
    def q_all(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
        return np.array([1.00, 0.90, 0.85, 0.95, 0.88])


def _base_ctx() -> ProspectContext:
    return sample_context({"sale_price": 230_000.0, "year_built": 1995.0}, np.random.default_rng(1))


@pytest.fixture
def client(settings: Settings) -> TestClient:
    orch = Orchestrator(
        policy=EpsilonGreedy(_FakeModel(), epsilon=settings.epsilon, rng=np.random.default_rng(0)),
        reward_model=_FakeModel(),
        distance_engine=HaversineEngine(speed_kmh=settings.walking_speed_kmh),
        store=EventStore(settings.db_path),
        settings=settings,
    )
    return TestClient(build_app(orch))


def test_recommend_feedback_route_roundtrip(client: TestClient) -> None:
    ctx_json = _base_ctx().model_dump(mode="json")

    rec = client.post("/recommend", json={"context": ctx_json})
    assert rec.status_code == 200
    body = rec.json()
    assert body["decision_id"]
    assert 0.0 < body["propensity"] <= 1.0
    assert set(body["q_values"]) == {a.value for a in ACTIONS}

    fb = client.post("/feedback", json={"decision_id": body["decision_id"], "outcome": "closed"})
    assert fb.status_code == 204

    base = _base_ctx()
    contexts = [
        base.model_copy(update={"address_id": f"near-{k}", "lat": 42.0 + 0.0005 * k, "lon": -93.6})
        for k in range(5)
    ]
    contexts.append(base.model_copy(update={"address_id": "far", "lat": 42.03, "lon": -93.6}))
    route = client.post("/route", json={"contexts": [c.model_dump(mode="json") for c in contexts]})
    assert route.status_code == 200
    rbody = route.json()
    assert "far" in rbody["dropped"]
    assert all(stop["order"] == i for i, stop in enumerate(rbody["stops"]))


def test_feedback_unknown_decision_returns_404(client: TestClient) -> None:
    resp = client.post("/feedback", json={"decision_id": "nope", "outcome": "info_given"})
    assert resp.status_code == 404


def test_malformed_recommend_returns_422(client: TestClient) -> None:
    resp = client.post("/recommend", json={"context": {"address_id": "x"}})
    assert resp.status_code == 422


def test_health_reports_policy_and_count(client: TestClient) -> None:
    before = client.get("/health").json()
    assert before["status"] == "ok"
    assert before["policy"] == "epsilon_greedy"
    assert before["decisions"] == 0

    client.post("/recommend", json={"context": _base_ctx().model_dump(mode="json")})
    after = client.get("/health").json()
    assert after["decisions"] == 1
