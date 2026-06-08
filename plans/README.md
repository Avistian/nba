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
  `pytest` green; every acceptance check in the phase file passes.

## How to use

Implement phases in dependency order (0 → 1 → 2 → {3,4,5} and 6 in parallel → 7 → 8). Within a
phase, write the listed tests first, then implement until green, then run the acceptance checks.
