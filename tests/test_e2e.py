"""System-level claims for the whole loop, verified on one small, seeded, offline shift.

These assert the PLAN.md promises end to end: the bandit beats uniform, its off-policy value beats
the logging baseline, average regret stays well under random, the router drops far outliers,
propensity is logged on every decision, and the API roundtrip works. The demo is run once
(module-scoped fixture) because it trains models; the API check is independent and fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_demo import DemoReport, _dense_block, run_demo  # noqa: E402


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> DemoReport:
    tmp = tmp_path_factory.mktemp("demo")
    settings = Settings(
        data_dir=tmp / "data",
        model_dir=tmp / "models",
        db_path=tmp / "events.db",
        n_bootstrap=2,
    )
    return run_demo(
        n_logs=2500,
        shift=24,
        replan_every=8,
        seed=7,
        settings=settings,
        ope_max_rows=400,
        write=False,
    )


def test_bandit_beats_uniform(demo: DemoReport) -> None:
    er = demo.expected_reward
    assert er["bandit"] > er["uniform"] * 1.1  # clear margin, not a coin flip


def test_selected_policy_beats_logging_baseline(demo: DemoReport) -> None:
    # The gate's promote rule (DR lower bound) is unit-tested in test_ope; here we assert the
    # selected policy's DR *value* dominates the logging baseline — it is genuinely better.
    selected_dr = demo.ope[demo.selected_policy]["dr"]
    assert selected_dr > demo.baseline_value
    # And the gate machinery produced a verdict + a confidence-aware lower bound.
    assert demo.selected_dr_lower_bound <= selected_dr


def test_regret_stays_well_below_uniform(demo: DemoReport) -> None:
    n = demo.n_decisions
    best = demo.expected_reward["oracle_best"]
    bandit_regret = (best - demo.expected_reward["bandit"]) / n
    uniform_regret = (best - demo.expected_reward["uniform"]) / n
    assert bandit_regret < uniform_regret
    # The bandit recovers most of the achievable reward (regret far below random's).
    assert bandit_regret < 0.8 * uniform_regret
    # Reported curve is well-formed (monotone non-decreasing cumulative regret).
    curve = demo.regret_curve
    assert len(curve) == n
    assert all(b >= a - 1e-9 for a, b in zip(curve, curve[1:], strict=False))


def test_propensity_logged_on_every_decision(demo: DemoReport) -> None:
    assert demo.n_decisions > 0
    assert demo.min_propensity > 0.0  # overlap holds => logs are OPE-valid


def test_router_drops_far_outliers(demo: DemoReport) -> None:
    assert demo.model is not None and demo.settings is not None
    block = _dense_block(12, demo.settings, seed=99, radius_km=0.25)
    # Inject two far, isolated doors ~33 km north — operationally absurd to walk to.
    outliers = [
        block[0].model_copy(update={"address_id": f"far-{k}", "lat": 42.33 + 0.01 * k})
        for k in range(2)
    ]
    contexts = block + outliers

    orch = Orchestrator(
        policy=EpsilonGreedy(demo.model, epsilon=0.1, rng=np.random.default_rng(0)),
        reward_model=demo.model,
        distance_engine=HaversineEngine(speed_kmh=demo.settings.walking_speed_kmh),
        store=EventStore(demo.settings.db_path),
        settings=demo.settings,
    )
    route = orch.plan_route(contexts)
    far_nodes = {len(block) + 1, len(block) + 2}  # 1-based door nodes for the two outliers
    assert far_nodes <= set(route.dropped)


# --------------------------------------------------------------------------------------------- #
# API roundtrip (independent, fast)
# --------------------------------------------------------------------------------------------- #
class _FakeModel:
    def q_all(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
        return np.array([1.00, 0.90, 0.85, 0.95, 0.88])


def test_api_recommend_feedback_route_roundtrip(settings: Settings) -> None:
    orch = Orchestrator(
        policy=EpsilonGreedy(_FakeModel(), epsilon=settings.epsilon, rng=np.random.default_rng(0)),
        reward_model=_FakeModel(),
        distance_engine=HaversineEngine(speed_kmh=settings.walking_speed_kmh),
        store=EventStore(settings.db_path),
        settings=settings,
    )
    client = TestClient(build_app(orch))
    base = sample_context({"sale_price": 230_000.0, "year_built": 1995.0}, np.random.default_rng(1))

    propensities: list[float] = []
    for _ in range(5):
        rec = client.post("/recommend", json={"context": base.model_dump(mode="json")})
        assert rec.status_code == 200
        body = rec.json()
        assert 0.0 < body["propensity"] <= 1.0
        propensities.append(body["propensity"])
        fb = client.post(
            "/feedback", json={"decision_id": body["decision_id"], "outcome": "info_given"}
        )
        assert fb.status_code == 204

    assert all(p > 0.0 for p in propensities)

    contexts = [
        base.model_copy(update={"address_id": f"near-{k}", "lat": 42.0 + 0.0005 * k, "lon": -93.6})
        for k in range(5)
    ]
    route = client.post("/route", json={"contexts": [c.model_dump(mode="json") for c in contexts]})
    assert route.status_code == 200
    assert route.json()["total_time_s"] >= 0.0
