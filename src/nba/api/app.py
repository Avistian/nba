"""FastAPI service over the orchestrator.

``build_app(orchestrator)`` is a factory so tests can inject a seeded/fake orchestrator without
touching disk models. The module-level ``app`` is the production instance: its lifespan loads
settings, the reward model, an epsilon-greedy logging policy, a Haversine engine, and the event
store, then assembles the orchestrator. Handlers are thin HTTP <-> orchestrator adapters.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response

from nba.api.models import (
    FeedbackRequest,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    RouteRequest,
    RouteResponse,
    RouteStop,
)
from nba.api.store import EventStore, UnknownDecisionError
from nba.pipeline.orchestrator import Orchestrator
from nba.routing.tsp_profits import Route
from nba.schema import ProspectContext


def _orchestrator(request: Request) -> Orchestrator:
    orchestrator: Orchestrator | None = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialized")
    return orchestrator


def _route_response(contexts: list[ProspectContext], route: Route) -> RouteResponse:
    """Map a solver :class:`Route` (depot at index 0) back to address-keyed stops."""
    stops: list[RouteStop] = []
    rank = 0
    for node in route.order:
        if node == 0:  # depot
            continue
        ctx = contexts[node - 1]
        stops.append(RouteStop(address_id=ctx.address_id, lat=ctx.lat, lon=ctx.lon, order=rank))
        rank += 1
    dropped = [contexts[node - 1].address_id for node in route.dropped]
    return RouteResponse(
        stops=stops,
        dropped=dropped,
        total_time_s=route.total_time_s,
        total_profit=route.total_profit,
    )


def build_app(
    orchestrator: Orchestrator | None = None,
    *,
    lifespan: object | None = None,
) -> FastAPI:
    """Build the FastAPI app, optionally with an injected orchestrator (for tests)."""
    app = FastAPI(title="NBA next-best-action service", lifespan=lifespan)  # type: ignore[arg-type]
    app.state.orchestrator = orchestrator

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest, request: Request) -> RecommendResponse:
        result = _orchestrator(request).recommend(req.context)
        return RecommendResponse(
            decision_id=result.decision_id,
            action=result.action,
            propensity=result.propensity,
            q_values=result.q_values,
        )

    @app.post("/feedback", status_code=204)
    def feedback(req: FeedbackRequest, request: Request) -> Response:
        try:
            _orchestrator(request).feedback(req.decision_id, req.outcome)
        except UnknownDecisionError as exc:
            raise HTTPException(status_code=404, detail="unknown decision_id") from exc
        return Response(status_code=204)

    @app.post("/route", response_model=RouteResponse)
    def route(req: RouteRequest, request: Request) -> RouteResponse:
        plan = _orchestrator(request).plan_route(req.contexts)
        return _route_response(req.contexts, plan)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        orchestrator = _orchestrator(request)
        return HealthResponse(
            status="ok",
            policy=orchestrator.policy_name,
            decisions=orchestrator.decision_count(),
        )

    return app


@asynccontextmanager
async def _default_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Assemble the production orchestrator from settings + persisted artifacts."""
    # Imported lazily so importing the module (e.g. in tests) never touches disk models.
    from nba.bandits.epsilon_greedy import EpsilonGreedy
    from nba.config import get_settings
    from nba.reward.model import RewardModel
    from nba.routing.distance import HaversineEngine

    settings = get_settings()
    settings.ensure_dirs()
    model = RewardModel.load(settings.model_dir)
    rng = np.random.default_rng(settings.seed)
    policy = EpsilonGreedy(model, epsilon=settings.epsilon, rng=rng)
    store = EventStore(settings.db_path)
    app.state.orchestrator = Orchestrator(
        policy=policy,
        reward_model=model,
        distance_engine=HaversineEngine(speed_kmh=settings.walking_speed_kmh),
        store=store,
        settings=settings,
    )
    try:
        yield
    finally:
        store.close()


app = build_app(lifespan=_default_lifespan)
