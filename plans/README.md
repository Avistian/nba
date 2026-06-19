# Implementation Plans — D2D Next Best Action (NBA)

Detailed, low-level, per-phase implementation plans derived from [../PLAN.md](../PLAN.md) and
grounded in [../docs/](../docs). Each file is self-contained: it lists exact files to create,
public signatures, behavioral contracts, edge cases, test matrix, and acceptance criteria so a
phase can be implemented (TDD) without re-deriving design.

## Phase index

| Phase | File | Depends on | Theme |
|-------|------|------------|-------|
| 0 | [phase-00-scaffold.md](phase-00-scaffold.md) | — | Project skeleton, tooling, config |
| 1 | [phase-01-schema.md](phase-01-schema.md) | 0 | Domain schema + reward function + features |
| 2 | [phase-02-simulator.md](phase-02-simulator.md) | 1 | D2D simulator + logged feedback |
| 3 | [phase-03-reward-model.md](phase-03-reward-model.md) | 2 | LightGBM `q(x,a)` + calibration |
| 4 | [phase-04-bandits.md](phase-04-bandits.md) | 3 | ε-greedy / UCB / Thompson behind `Policy` |
| 5 | [phase-05-ope.md](phase-05-ope.md) | 4 | IPS/DM/DR estimators + promotion gate |
| 6 | [phase-06-routing.md](phase-06-routing.md) | 1 (∥ 3–5) | Haversine + OR-Tools TSP-with-Profits |
| 7 | [phase-07-orchestrator-api.md](phase-07-orchestrator-api.md) | 4, 6 | Orchestrator + FastAPI + event store |
| 8 | [phase-08-demo-verification.md](phase-08-demo-verification.md) | 7 | Full-shift demo + cross-module verification |
| 9 | [phase-09-relational-dataset.md](phase-09-relational-dataset.md) | 1, 2 | Relational dataset (mirrors flat) + graph builder |
| 10 | [phase-10-orienteering.md](phase-10-orienteering.md) | 6 | Upgrade 1: time budget + team + road-aware (OSRM) |
| 11 | [phase-11-risk-aware-routing.md](phase-11-risk-aware-routing.md) | 4, 7 | Upgrade 3: risk-aware door pricing (mean − κ·std) |
| 12 | [phase-12-decision-focused-learning.md](phase-12-decision-focused-learning.md) | 3, 5, 6 | Upgrade 2: decision-focused learning (SPO+) |
| 13 | [phase-13-dynamic-stochastic-routing.md](phase-13-dynamic-stochastic-routing.md) | 7, 11 | Upgrade 5: stochastic prizes + lookahead replan |
| 14 | [phase-14-relational-deep-learning.md](phase-14-relational-deep-learning.md) | 9, 3, 5 | RDL value model (GNN behind `QModel`) |
| 15 | [phase-15-neural-combinatorial-optimization.md](phase-15-neural-combinatorial-optimization.md) | 10 (∥ 14) | Upgrade 4: neural CO router (deferred) |
| 16 | [phase-16-decision-focused-rdl.md](phase-16-decision-focused-rdl.md) | 12, 14 | Decision-focused RDL fusion (deferred) |
| 17 | [phase-17-experiment-leaderboard.md](phase-17-experiment-leaderboard.md) | 9, 5, 8 (after Phase 9, before upgrades) | Append-only leaderboard: lift/regression per experiment |
| 18 | [phase-18-drift-monitoring-retrain-loop.md](phase-18-drift-monitoring-retrain-loop.md) | 3, 5, 7, 8, 17 (after Phase 17, before/up parallel with upgrades) | Drift signals, monitor, conditional retrain + DR gate, drift sim demo |

Phases 9–16 are the improvement roadmap from [../docs/11](../docs/11-improving-nba-spatio-relational-optimization.md)
and [../docs/12](../docs/12-relational-deep-learning-mixin.md): each is **feature-flagged, off by
default**, and preserves every rail. The relational dataset (Phase 9) is built first as a new dataset
that mirrors the flat one. Phase 17 is the **evaluation harness** that grades every other phase as a
**lift, regression, or neutral** on an append-only leaderboard — built **right after Phase 9** (so it
can grade on both datasets) and **before the upgrades**, each of which must be tested and prove its
value here (a regression blocks adoption). Phase 18 closes the **operational ML loop** (monitor drift,
retrain only when triggered, promote through the same DR gate). Build order: **9 → 17 → 18 → 10–16**.

## Architecture through-line

```
context x → reward model q(x,a) → bandit policy (explore) → per-door profit → TSP-P walkable route
```

Every recommendation logs its propensity `p`; logs feed the OPE gate that must pass before any
policy is promoted. **The bandit proposes, the router disposes.**

## Conventions (apply to every phase)

- **Runtime/tooling:** Python 3.11+, `uv` for deps, `ruff` for lint/format, `pyright` for types,
  `pytest` (+ `pytest-cov`) for tests.
- **Layout:** src-layout under `src/nba/`; tests under `tests/`; one-off entrypoints in `scripts/`.
- **Boundaries:** pydantic models at I/O boundaries (API, file load); `Protocol`s for swappable
  components (`Policy`, `DistanceEngine`); plain dataclasses/numpy internally for hot paths.
- **Determinism:** every stochastic component accepts an explicit `seed`/`numpy.random.Generator`;
  tests pin seeds and assert reproducibility.
- **Ethics:** features pass through an allow-list (`features.ALLOWED_FEATURES`); no protected
  attributes; exploration capped in sensitive contexts (config flag).
- **Oracle hygiene:** the simulator's `true_reward` is the *only* ground-truth oracle; it must
  never leak into features, the reward model, or any policy.
- **Definition of Done (per phase):** code + tests written TDD-style; `ruff` and `pyright` clean;
  `pytest` green; every acceptance check in the phase file passes. **For Phases 10–16 (each upgrade):
  additionally a logged Phase 17 leaderboard row vs `baseline` that is a lift or a documented neutral
  — a regression blocks adoption.**

## How to use

Implement phases in dependency order (0 → 1 → 2 → {3,4,5} and 6 in parallel → 7 → 8). Within a
phase, write the listed tests first, then implement until green, then run the acceptance checks.
