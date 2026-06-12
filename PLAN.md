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

## Resolved decisions

1. Dependency manager: **uv**.
2. Event store: **SQLite**, append-only (no `UPDATE`/`DELETE`; corrections are new rows).
3. Thompson uncertainty: **bootstrap LightGBM ensemble** (reuses the reward model).
4. Ethics: **feature allow-list** (no protected/geo fields) + a **sensitive-context exploration
   cap** (`EthicalPolicy`) that preserves full support so logs stay OPE-valid.

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
`scripts/run_demo.py` simulates a full shift end-to-end (logs → reward model → OPE gate → walk the
route → compare vs baselines → report), writing `artifacts/demo_report.json`. `tests/test_e2e.py`
asserts the system-level claims and `tests/test_ethics.py` the guardrails; `src/nba/ethics.py` adds
the sensitive-context exploration cap.

## Verification matrix

| Claim | Where | State |
|-------|-------|-------|
| `pytest` green across all modules | `make test` (143 passed, 1 skipped) | ✅ |
| OPE estimators match OBP within tolerance | `tests/test_ope.py` (slow) | ✅ |
| Bandit beats the uniform baseline | `tests/test_e2e.py::test_bandit_beats_uniform` | ✅ |
| Selected policy's value beats the logging baseline | `tests/test_e2e.py::test_selected_policy_beats_logging_baseline` | ✅ |
| Regret stays far below random (near-optimal) † | `tests/test_e2e.py::test_regret_stays_well_below_uniform` | ✅ |
| TSP-P drops far-flung outliers, respects windows/capacity | `tests/test_e2e.py`, `tests/test_routing.py` | ✅ |
| API `recommend → feedback → route` roundtrip | `tests/test_e2e.py`, `tests/test_api.py` | ✅ |
| Propensity present on every recommend | `tests/test_e2e.py::test_propensity_logged_on_every_decision` | ✅ |
| Event log append-only | `tests/test_store.py` | ✅ |
| No protected/geo attributes in features | `tests/test_ethics.py` | ✅ |
| Exploration capped in sensitive contexts | `tests/test_ethics.py` | ✅ |
| No oracle leakage into learning modules | `tests/test_ethics.py::test_no_oracle_leak` | ✅ |

† The PLAN originally framed this as "cumulative regret trends down." That downward *curve* is an
**online-learning** phenomenon (the model improving across many rounds). A single deployed shift
runs a **fixed**, already-gated policy, so its per-round regret is *stationary*; the verifiable,
meaningful claim is that this regret sits far below a uniform-random policy's (the bandit is close
to optimal). The demo reports the full curve so the stationarity is visible.

## Status

**All phases (0–8) are implemented and verified** — `ruff`/`pyright` clean, `pytest` green
(143 passed, 1 skipped). The full loop — simulator → reward model → bandit policies → OPE gate →
TSP-with-profits router → orchestrator + FastAPI service over an append-only SQLite event store,
with ethics guardrails — runs offline and end-to-end. `make demo` runs it for one shift and writes
`artifacts/demo_report.json`. Each phase has a mirroring notebook in [notebooks/](notebooks), and
[docs/09-build-nba-from-scratch.md](docs/09-build-nba-from-scratch.md) explains the whole build from
first principles.
