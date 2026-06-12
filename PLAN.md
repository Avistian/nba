# Implementation Plan — D2D Next Best Action (NBA) Prototype

> Grounded in [docs/](docs) (esp. [02-concepts-to-learn.md](docs/02-concepts-to-learn.md),
> [03-data.md](docs/03-data.md), [05-implementation-steps.md](docs/05-implementation-steps.md),
> [08-bandits-and-offline-evaluation.md](docs/08-bandits-and-offline-evaluation.md)).

## Goal

Build a runnable, offline-first Python implementation of the NBA system: a
**reward model → contextual bandit → OPE gate → TSP-P router** loop, exposed through a thin
FastAPI service. A built-in D2D simulator generates logged bandit feedback `(x, a, r, p)` with
known ground truth, so the whole pipeline — including offline policy evaluation — is testable
without real field data.

## Architecture (the through-line)

```
context x → reward model q(x,a) → bandit policy (explore) → per-door profit → TSP-P walkable route
```

Every recommendation logs its propensity `p`; logs feed the OPE gate that must pass before any
policy is promoted. **The bandit proposes, the router disposes.**

## Decisions (locked)

| Topic | Decision |
|-------|----------|
| Scope | Local Python prototype **+ thin FastAPI service** (`/recommend`, `/feedback`, `/route`). AWS deferred. |
| Data | Build a **D2D simulator** (ground-truth rewards) + **Open Bandit Dataset** for OPE validation + **Ames/ACS** for reward features. |
| Bandit | Ship **all three** policies (ε-greedy, UCB, Thompson) behind one `Policy` protocol. |
| Routing | **Haversine** distance matrix first; `DistanceEngine` interface so OSRM/Valhalla can drop in later. |

## Open follow-ups (defaults chosen if unanswered)

1. Dependency manager: **uv** (default) / pip+venv / poetry.
2. Event store: **SQLite** (default, queryable) / parquet append.
3. Thompson uncertainty: **bootstrap LightGBM ensemble** (default, reuses reward model) / Bayesian linear head.

## Package layout (to create)

```
nba/
  pyproject.toml, README.md, Makefile
  src/nba/
    schema.py        # Action enum, ProspectContext, BanditEvent, REWARD map
    config.py        # pydantic settings
    data/    simulator.py, ames.py, obd.py, features.py
    reward/  model.py        # LightGBM q(x,a) + calibration
    bandits/ base.py, epsilon_greedy.py, ucb.py, thompson.py
    ope/     estimators.py   # IPS / DM / DR
             gate.py
    routing/ distance.py, territories.py, tsp_profits.py
    pipeline/ orchestrator.py
    api/     app.py          # FastAPI
             store.py        # append-only event log
  tests/   test_*.py per module
  scripts/ generate_logs.py, train_reward.py, evaluate_policy.py, run_demo.py
```

## Phases

### Phase 0 — Scaffold
`pyproject.toml` (lightgbm, ortools, obp, fastapi, uvicorn, pydantic, pandas, numpy,
scikit-learn, scipy, pytest), package skeleton, `config.py`, `Makefile`.

### Phase 1 — Schema + reward function *(depends 0)*
`Action` enum (`KNOCK_NOW`, `LEAVE_FLYER`, `SKIP_DOOR`, `PITCH_SOLAR`, `PITCH_SECURITY`),
`ProspectContext` (prospect + environment + spatial), `BanditEvent(x, a, r, p, address_id, ts,
lat, lon)`, `REWARD` map, `featurize(context, action)`. Mirrors doc 03 §3.1.

### Phase 2 — D2D simulator + feature substrate *(depends 1)*
Latent conversion function over context × action × time → ground-truth reward probabilities; a
stochastic behavior policy that **logs propensity p**; Ames/ACS draws for realistic context;
emits logged-feedback parquet.

### Phase 3 — Reward model *(depends 2)*
LightGBM `q(x,a) = E[r|x,a]` on logs; isotonic calibration; exploitation baseline (argmax) for
comparison. Doc 05 Step 2.

### Phase 4 — Bandit policies (pluggable) *(depends 3)*
`Policy` protocol → `recommend(ctx, actions) -> (action, p)` + `action_dist(ctx)`. Implement
**ε-greedy**, **UCB** (count/ensemble uncertainty), **Thompson** (bootstrap LightGBM ensemble).
Docs 02 §2.2, 08 §8.3.

### Phase 5 — OPE + promotion gate *(depends 4)*
IPS, DM, DR estimators; **validate against OBP** on the Open Bandit Dataset (relative-ee within
tolerance); gate promotes a candidate only if its estimated value beats the logging baseline
within a confidence bound. Doc 08 §8.4.

### Phase 6 — Routing / TSP-P *(parallel with 3–5)*
Haversine matrix + `DistanceEngine` interface (OSRM stub); KMeans walkable territories; OR-Tools
TSP-P with `AddDisjunction` drop-penalty = door profit, plus time-window + capacity constraints.
Doc 05 Step 5.

### Phase 7 — Orchestrator + FastAPI *(depends 4, 6)*
Orchestrator wires bandit profits → TSP-P. Endpoints: `/recommend` (logs p), `/feedback`
(appends reward), `/route` (re-solve). Append-only SQLite/parquet store.

### Phase 8 — Demo + tests + verification
`run_demo.py` simulates a full shift end-to-end; pytest per module.

## Verification

- `pytest` green across all modules.
- OPE estimators match OBP ground-truth within tolerance on the Open Bandit Dataset.
- Simulator: bandit beats the uniform baseline; cumulative regret trends down over rounds.
- TSP-P returns a walkable subset, drops far-flung outliers, respects time windows + capacity.
- API smoke test: `recommend → feedback → route` roundtrip; **propensity present on every
  recommend**; event log is append-only.
- Ethics guardrails: no protected attributes in features; exploration capped in sensitive
  contexts.

## Status

Phases 0–7 are implemented and verified (`ruff`/`pyright` clean, `pytest` green). The full loop —
simulator → reward model → bandit policies → OPE gate → TSP-with-profits router → orchestrator +
FastAPI service over an append-only SQLite event store — runs offline and end-to-end. Each phase
has a mirroring notebook in [notebooks/](notebooks). **Remaining:** Phase 8 (a `run_demo.py` that
simulates a full shift end-to-end).
