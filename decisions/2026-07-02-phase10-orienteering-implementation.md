## Decision: Phase 10 orienteering ships as additive solver params with an OSRM engine over stdlib urllib, an API that stays single-vehicle, and team wall-clock time modeled as the slowest rep

## Context

Phase 10 (Upgrade 1) makes the router state the real Orienteering Problem: a global shift-time budget
(OP), multiple reps (Team Orienteering / TOP), and real road travel times (`OSRMEngine`). It must stay
byte-identical to today's single-vehicle, window-only, straight-line behavior until a flag is set, and
must not regress the verified 0-8 loop. Several implementation forks had to be resolved.

## Alternatives considered

- **OSRM HTTP client:** `requests`/`httpx` (a new runtime dependency) vs stdlib `urllib.request`.
  `httpx` exists only in the dev group (for the FastAPI TestClient), so importing it from `src/` would
  either add a runtime dep or create a dev-only import in production code.
- **Multi-vehicle return type:** always return `list[Route]` vs return a single `Route` when
  `num_vehicles == 1` and a list otherwise. Always-list would touch every existing caller/test.
- **API `/route` with a team plan:** change `RouteResponse` to carry per-rep routes vs keep the
  single-vehicle response and report the first rep's route.
- **Team wall-clock time in the graded demo:** sum rep times vs take the max (reps walk in parallel).
- **Budget test strictness:** assert `total_time_s <= budget` exactly vs allow a small tolerance.

## Reasoning

- **OSRM via stdlib `urllib.request`** keeps runtime dependencies unchanged and CI fully offline. The
  single network call sits in a private `OSRMEngine._fetch` seam that tests monkeypatch; failures wrap
  into the existing `RoutingError`. Durations are symmetrized (`0.5*(m+m.T)`) and zero-diagonaled to
  satisfy the `DistanceEngine` contract that OR-Tools relies on.
- **`Route | list[Route]` (single when `num_vehicles == 1`)** preserves the Phase 6 API exactly, so all
  existing routing/orchestrator/e2e tests pass verbatim; only the team path returns a list. A single
  `AddDisjunction` per door guarantees no double-serving; `dropped` is the global no-rep-served set.
- **API stays single-vehicle** in this phase (no schema change); if a team config is set, `/route`
  reports the first rep's route. Team routing is exercised through the solver + graded demo/leaderboard.
- **Team time = max rep time, throughput = union of served doors** because reps walk concurrently, so
  the shift's wall-clock cost is the slowest rep and value collected is the union.
- **Budget assertion uses a small rounding tolerance** (`<= budget + len(order)` seconds): OR-Tools
  caps the integer `Time` dimension at the budget, while `total_time_s` is a float sum, so per-arc
  integer rounding can differ by well under a second per arc.

## Budget frame bug found + fixed (span vs absolute cap)

The doc-11 recipe `time_dim.CumulVar(routing.End(v)).SetMax(B)` assumes the Time clock starts at 0.
But `plan_route` uses **seconds-of-day** time windows (16:00-19:00 => cumul in 57600..68400), so an
absolute `End <= 28800` cap is contradictory with the door windows and the whole solve went
infeasible (`RoutingError`) in the first leaderboard run. Fix: bound each rep's **span**
(`SetSpanUpperBoundForVehicle(B, v)` = `end - start`), which is frame-independent and reduces to
`end <= B` in the no-windows case. Regression-guarded by
`tests/test_routing.py::test_time_budget_composes_with_seconds_of_day_windows`.

## Leaderboard results (6 shifts x seed 7, n_logs 3000, shift 40, vs `baseline`)

| experiment | realized value | Δ value | verdict |
|---|---|---|---|
| baseline | +4.526 | +0.000 | neutral (reference) |
| phase10-budget (`NBA_USE_TIME_BUDGET=1`) | +4.526 | +0.000 | **neutral** |
| phase10-team2 (`NBA_NUM_VEHICLES=2`) | +4.206 | -0.321 | **regression** |

- **Budget => neutral (byte-identical value):** the 3-hour residential window is tighter than the
  8-hour budget, so the budget never binds on the demo block. It is a feasibility guarantee, not a
  value lever, at this scale — a deliberate, documented neutral.
- **Team => regression at single-block scale:** on one 0.3 km walkable block a lone rep already
  services every economically-worthwhile door within the window, so a second rep is redundant and its
  different traversal only adds sampling noise (-7%, within run-to-run variance at 6 shifts). Team
  Orienteering's value target is **multi-territory** routing, which the single-block demo geometry
  cannot exercise. Adoption stays **off by default** (the regression correctly blocks default-on),
  and the *mechanics* are proven by unit tests (budget binds and drops doors as it tightens; a team
  plan partitions doors across reps with no double-serving).
- **OSRM => deferred:** `phase10-osrm` needs a reachable OSRM `/table` service (the engine makes real
  HTTP calls); it is not run in this offline environment. The engine is validated by mocked-HTTP unit
  tests instead.

## Trade-offs accepted

- No live OSRM validation in CI (mocked seam only); the `phase10-osrm` leaderboard row is deferred
  until a reachable OSRM `/table` service is available.
- The budget/team value story is scale-dependent; the single-block demo shows neutral/regression, so
  the flags remain opt-in and the unit tests (not the board) are the proof of correct mechanics.

## Supersedes

None. First Phase 10 decision; builds on the Phase 6 routing design.
