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
5. **Improvement roadmap is feature-flag-first.** Every upgrade in Phases 9–16 (the
   [docs/11](docs/11-improving-nba-spatio-relational-optimization.md) /
   [docs/12](docs/12-relational-deep-learning-mixin.md) work) ships behind an `NBA_*` flag that
   **defaults to today's behavior**, so the verified 0–8 loop is untouched until a flag is set.
6. **The relational dataset mirrors the flat one.** It is added as a *new* dataset
   (`dataset_mode="relational"`) emitting a schema-identical `BanditEvent` stream, not a rewrite of
   the existing simulator — so RDL can be benchmarked head-to-head against LightGBM and abandoned
   cleanly if it doesn't win the OPE gate.

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

## Improvement phases (planned, feature-flagged)

Phases 9–16 implement the upgrade roadmap from
[docs/11](docs/11-improving-nba-spatio-relational-optimization.md) and
[docs/12](docs/12-relational-deep-learning-mixin.md). Each is **off by default** behind an `NBA_*`
flag and preserves every rail (no oracle leak, ethics allow-list, calibration, DR promotion gate).
Per-phase specs live in [plans/](plans); step-by-step build docs in [docs/](docs).

**Build order: relational dataset → leaderboard → upgrades.** The relational dataset (Phase 9) is
built first; the experiment leaderboard (Phase 17) is built **right after it** so it can grade
experiments on both the flat and relational datasets; then each upgrade (Phases 10-16) must be tested
**and prove its value** on that leaderboard before adoption. (Phase numbers are stable IDs — like
Phase 6 "parallel with 3-5", build order follows dependencies, not the file number.)

**Every upgrade must be tested and prove itself on the leaderboard.** Phase 17 adds an append-only
experiment leaderboard ([plan](plans/phase-17-experiment-leaderboard.md) ·
[doc](docs/21-experiment-leaderboard.md)): each flag config is run against the `baseline` (all flags
off = today's pipeline) and judged a **lift, regression, or neutral** on a common metric set. A
**lift requires both a higher primary metric (realized shift value) and clearing the same DR gate** —
so wins can't be noise — and a **regression blocks the upgrade's adoption**. Each phase below names
its leaderboard experiment(s) in its plan file, and is not "done" until it has passing tests **and** a
logged leaderboard row that is a lift or a deliberate, documented neutral.

### Phase 9 — Relational dataset *(depends 1, 2)* — [plan](plans/phase-09-relational-dataset.md) · [doc](docs/13-relational-dataset.md)
A **new dataset that mirrors** the flat simulator (`dataset_mode`) but with genuine relational/
temporal ground truth (households, neighbor/competitor edges, interaction histories) and a
heterogeneous-graph builder behind a graph allow-list. Emits a schema-identical `BanditEvent` stream
so all downstream code is unchanged. **Foundation for RDL.**

### Phase 10 — Upgrade 1: Orienteering *(depends 6)* — [plan](plans/phase-10-orienteering.md) · [doc](docs/14-orienteering-upgrade.md)
Explicit shift-time budget (OP), multi-rep routing (TOP), and a real `OSRMEngine` — additive params
on `solve_tsp_profits`, flags `use_time_budget`/`shift_hours`, `num_vehicles`, `distance_engine`.

### Phase 11 — Upgrade 3: Risk-aware routing *(depends 4, 7)* — [plan](plans/phase-11-risk-aware-routing.md) · [doc](docs/15-risk-aware-routing.md)
Price doors `mean − κ·std` over the bootstrap ensemble (optional CVaR); flags
`use_risk_aware_routing`, `risk_kappa` (`κ=0` is a no-op).

### Phase 12 — Upgrade 2: Decision-focused learning *(depends 3, 5, 6)* — [plan](plans/phase-12-decision-focused-learning.md) · [doc](docs/16-decision-focused-learning.md)
Train the reward model on **route value** not prediction error: decision-aware reweighting, then an
SPO+ fine-tune behind the `QModel` protocol; flags `use_decision_focused`, `df_mode`.

### Phase 13 — Upgrade 5: Dynamic/stochastic routing *(depends 7, 11)* — [plan](plans/phase-13-dynamic-stochastic-routing.md) · [doc](docs/17-dynamic-stochastic-routing.md)
Stochastic prizes + optional lookahead/rollout replanning on top of `replan`; flags
`use_stochastic_prizes`, `use_lookahead`.

### Phase 14 — RDL value model *(depends 9, 3, 5)* — [plan](plans/phase-14-relational-deep-learning.md) · [doc](docs/18-relational-deep-learning.md)
A calibrated R-GCN/GraphSAGE `q(x,a)` behind the `QModel` protocol over the Phase 9 graph (optional
`rdl` extra); flag `reward_model_kind`. Promotes only if it beats LightGBM through the same DR gate.

### Phase 15 — Upgrade 4: Neural CO *(deferred)* — [plan](plans/phase-15-neural-combinatorial-optimization.md) · [doc](docs/19-neural-combinatorial-optimization.md)
Attention encoder-decoder router behind a `Router` protocol with OR-Tools as the test oracle; flag
`router_kind`. Build only at fleet scale.

### Phase 16 — Decision-focused RDL *(deferred)* — [plan](plans/phase-16-decision-focused-rdl.md) · [doc](docs/20-decision-focused-rdl.md)
The research frontier: train the GNN end-to-end through the orienteering optimizer (fuses Phases 12 +
14); flag `use_decision_focused_rdl`.

### Phase 17 — Experiment leaderboard *(depends 9, 5, 8; built right after Phase 9, before the upgrades)* — [plan](plans/phase-17-experiment-leaderboard.md) · [doc](docs/21-experiment-leaderboard.md)
The cross-cutting evaluation harness for Phases 10–16: `src/nba/eval/{metrics,leaderboard}.py` +
`scripts/run_experiment.py` write an **append-only** `artifacts/leaderboard.jsonl` recording each
experiment's flags, metrics, per-metric delta vs `baseline`, DR-gate result, and **lift/regression/
neutral** verdict. Built immediately after the relational dataset (so it can grade on both datasets)
and before any upgrade, which must each prove value here.

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
| Relational dataset mirrors flat (schema round-trip; degenerate==flat) | `tests/test_relational_simulator.py` (Phase 9) | ⏳ planned |
| Graph allow-list blocks geo/identity/protected at node+edge | `tests/test_graph.py` (Phase 9) | ⏳ planned |
| Budgeted route ≤ shift budget; team route never double-serves | `tests/test_routing.py` (Phase 10) | ⏳ planned |
| All upgrade flags off reproduce today's behavior exactly | per-phase tests (Phases 9–16) | ⏳ planned |
| Risk-aware routing reduces realized-value variance (κ=0 no-op) | `tests/test_orchestrator.py` (Phase 11) | ⏳ planned |
| Decision-focused model lowers decision regret at ≥ OPE | `tests/test_decision_focused.py` (Phase 12) | ⏳ planned |
| Stochastic prizes shrink downside risk | `tests/test_dynamic.py` (Phase 13) | ⏳ planned |
| RDL model promotes only via the same DR gate on route value | `tests/test_graph_model.py` (Phase 14) | ⏳ planned |
| Every phase logs a lift/regression leaderboard row vs baseline | `tests/test_leaderboard.py` (Phase 17) | ⏳ planned |
| Leaderboard is append-only; lift requires primary-metric gain + DR gate | `tests/test_leaderboard.py` (Phase 17) | ⏳ planned |

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
