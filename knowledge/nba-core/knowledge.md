# NBA Core — Knowledge

Facts and patterns about the system loop, orchestration, and persistence.

## The through-line

```
context x → reward model q(x,a) → bandit policy (explore) → per-door profit → TSP-P walkable route
```

Every decision logs `(context, action, propensity)`; outcomes append later. Logs feed retraining and
OPE. **The bandit proposes, the router disposes.**

## Module seams

| Seam | Module | Role |
|------|--------|------|
| Domain types | `schema.py` | `Action`, `Outcome`, `REWARD`, `ProspectContext`, `BanditEvent` |
| Config | `config.py` | `Settings` via `NBA_*` env vars; single `seed` |
| Orchestration | `pipeline/orchestrator.py` | `recommend`, `feedback`, `plan_route`, `replan` |
| HTTP edge | `api/app.py` | Thin FastAPI over orchestrator; `build_app` factory for tests |
| Persistence | `api/store.py` | Append-only SQLite `EventStore` |

## Orchestrator behavior

- `recommend(ctx)` → policy picks `(action, propensity)` → decision row appended → `RecommendResult`
  with `q_values`.
- `feedback(decision_id, outcome)` → outcome row appended (unknown id → error / HTTP 404).
- `door_profit(ctx)` = Σ_a π(a|x)·q(x,a) by default; `argmax_profit=True` uses greedy max q.
- `plan_route(contexts)` → depot at centroid, TSP-P under capacity + time windows.

## Event store invariants

- **Append-only**: no `UPDATE`/`DELETE` on decisions or outcomes.
- Corrections = new outcome rows; readers take latest by autoincrement id.
- Full `ProspectContext` stored as JSON for faithful `load_events()` → `BanditEvent` replay.
- `ingest_bandit_events` skips decisions and outcomes that already exist so bulk replay is idempotent.

## Determinism

- `Settings.seed` + explicit `np.random.Generator` instances everywhere stochastic.
- TSP solver: fixed inputs + time limit + single thread → reproducible routes.
- Tests pin `NBA_SEED=7` via `conftest.py`.

## Status (as of Phase 8)

All eight phases implemented. `make demo` runs full offline shift; `make check` = ruff + pyright +
pytest (143 passed, 1 skipped).
