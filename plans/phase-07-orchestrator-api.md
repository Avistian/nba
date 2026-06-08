# Phase 7 — Orchestrator + FastAPI + event store

**Depends on:** Phases 4 and 6. **Goal:** wire the loop together — bandit profits feed the
TSP-P router — and expose it as a thin FastAPI service backed by an **append-only** event store
that records every decision's propensity (so logs can later feed the OPE gate).

## Files to create

```
src/nba/api/__init__.py
src/nba/api/store.py
src/nba/api/models.py
src/nba/api/app.py
src/nba/pipeline/__init__.py
src/nba/pipeline/orchestrator.py
tests/test_store.py
tests/test_orchestrator.py
tests/test_api.py
```

## `src/nba/api/store.py` — append-only `EventStore`

```python
class EventStore:
    """SQLite, append-only. Two tables joined by decision_id; outcomes inserted, never updated.
       Schema:
         decisions(decision_id PK, ts, address_id, lat, lon, context_json,
                   action, propensity, policy_name)
         outcomes (id PK autoinc, decision_id FK, ts, outcome, reward)   -- 1:N append; latest wins
    """
    def __init__(self, db_path: Path): ...           # PRAGMA journal_mode=WAL; create tables IF NOT EXISTS
    def append_decision(self, *, context, action, propensity, policy_name) -> str:  # returns decision_id (uuid4)
    def append_outcome(self, decision_id: str, outcome: Outcome) -> None:           # reward = reward_for(outcome)
    def load_events(self) -> list[BanditEvent]:      # JOIN latest outcome per decision → BanditEvent
    def decision_count(self) -> int; def outcome_count(self) -> int
```

- **Append-only invariant:** no `UPDATE`/`DELETE`. A correction is a *new* outcome row; readers
  take the latest by `id`. This preserves the audit trail and keeps the log honest for OPE.
- Stores the full `context_json` (pydantic `model_dump_json`) so `load_events` reconstructs exact
  `ProspectContext`s for training/OPE.
- `propensity` is `NOT NULL` at the DB level — schema enforces "every decision logs p".

## `src/nba/pipeline/orchestrator.py`

```python
class Orchestrator:
    def __init__(self, *, policy: Policy, reward_model: RewardModel,
                 distance_engine: DistanceEngine, store: EventStore, settings: Settings): ...

    def recommend(self, ctx: ProspectContext) -> RecommendResult:
        # action, p = policy.recommend(ctx); decision_id = store.append_decision(...)
        # return RecommendResult(decision_id, action, propensity=p, q_values=reward_model.q_all(ctx))

    def feedback(self, decision_id: str, outcome: Outcome) -> None:
        # store.append_outcome(decision_id, outcome)

    def plan_route(self, contexts: list[ProspectContext]) -> Route:
        # per door: profit = max over actions of expected reward under policy.action_dist:
        #     profit_d = Σ_a action_dist(ctx_d)[a] * reward_model.q(ctx_d, a)  (bandit-weighted value)
        #   (depot prepended at rep's current location, profit 0)
        # coords = [depot] + [(lat,lon) per door]; tm = distance_engine.time_matrix(coords)
        # tw = residential window (settings.time_window → seconds-of-day) per door
        # return solve_tsp_profits(coords, profits, tm, capacity=settings.shift_capacity, ...)

    def replan(self, remaining: list[ProspectContext]) -> Route:   # plan_route over not-yet-visited
```

- **The seam:** `profit_d` is the bandit's *expected* value of the door (probability-weighted over
  its own action distribution), not a raw argmax — so exploration value flows into routing. Document
  this choice; alternative (argmax `q`) noted as a config toggle.
- Orchestrator is pure-Python and DI-constructed; the API layer only adapts HTTP ↔ these methods.

## `src/nba/api/models.py` (pydantic request/response)

```python
class RecommendRequest(BaseModel):  context: ProspectContext
class RecommendResponse(BaseModel): decision_id: str; action: Action; propensity: float
                                    q_values: dict[Action, float]
class FeedbackRequest(BaseModel):   decision_id: str; outcome: Outcome
class RouteRequest(BaseModel):      contexts: list[ProspectContext]
class RouteStop(BaseModel):         address_id: str; lat: float; lon: float; order: int
class RouteResponse(BaseModel):     stops: list[RouteStop]; dropped: list[str]
                                    total_time_s: float; total_profit: float
class HealthResponse(BaseModel):    status: Literal["ok"]; policy: str; decisions: int
```

## `src/nba/api/app.py`

```python
def build_app(orchestrator: Orchestrator) -> FastAPI   # factory for tests (DI the orchestrator)

# default app: lifespan loads Settings → RewardModel.load → policy → HaversineEngine → EventStore
app = FastAPI(lifespan=...)
@app.post("/recommend",  response_model=RecommendResponse)   # logs propensity, returns decision_id
@app.post("/feedback",   status_code=204)                    # append outcome
@app.post("/route",      response_model=RouteResponse)       # bandit profits → TSP-P
@app.get ("/health",     response_model=HealthResponse)
```

- `build_app(orchestrator)` factory lets `TestClient` inject a fake/seeded orchestrator without
  touching disk models. Errors → proper HTTP codes (404 unknown `decision_id`, 422 validation).

## Tests

`tests/test_store.py`
- `append_decision` returns a uuid; `decision_count` increments; `propensity` persisted and > 0.
- `append_outcome` then `load_events` → a `BanditEvent` with reward `== reward_for(outcome)`.
- **Append-only:** two outcomes for one decision → both rows exist; `load_events` takes the latest;
  no UPDATE/DELETE issued (assert via a SQL trace/spy or by row counts).
- Round-trip context fidelity: loaded `ProspectContext` equals the stored one.

`tests/test_orchestrator.py`
- `recommend` writes exactly one decision and returns a non-null propensity + decision_id.
- `plan_route` profits equal the bandit-weighted `q` expectation (checked against a hand
  computation on a tiny fixture); high-value cluster kept, far cheap door dropped.
- `feedback` followed by `load_events` yields a labeled event usable by Phase 3/5.

`tests/test_api.py` (FastAPI `TestClient` on `build_app(fake_orchestrator)`)
- **Roundtrip:** `/recommend` → `/feedback` → `/route` all 2xx; every `/recommend` response has a
  non-null `propensity` and `decision_id`.
- `/feedback` with unknown decision_id → 404; malformed body → 422.
- `/health` returns policy name and current decision count.

## Acceptance

- Every `/recommend` logs a decision with `propensity > 0` and returns its `decision_id`; the store
  is provably append-only.
- `/route` returns an ordered walkable plan with dropped outliers, driven by bandit-weighted door
  profit.
- TestClient `recommend→feedback→route` roundtrip passes; `ruff`/`pyright` clean; `pytest` green.
