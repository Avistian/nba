# Phase 10 — Upgrade 1: Budgeted, team, road-aware orienteering

**Depends on:** Phase 6 (routing). **Goal:** make the optimizer state and solve the *real* problem —
the **Orienteering Problem (OP)** — by adding an explicit shift-time budget, multiple reps (Team
Orienteering), and real road travel times, all behind feature flags that default to today's
single-vehicle, window-only PCTSP behavior. This is the cheapest, surest value in the roadmap and
needs no machine learning. Grounded in
[docs/11 §3-4](../docs/11-improving-nba-spatio-relational-optimization.md) and the build in
[docs/14](../docs/14-orienteering-upgrade.md).

> **Where today's code sits** (doc 11 §3.3): `solve_tsp_profits` is already a single-vehicle
> Prize-Collecting TSP with per-node time windows + capacity. The gaps are (a) a global time budget,
> (b) multiple vehicles, (c) real road times (`OSRMEngine` is a stub). This phase closes all three
> additively.

## Feature flags (added to `src/nba/config.py` `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `use_time_budget` | `bool = False` | Bound the route's end-of-day cumulative `Time` var (OP). Off => today's behavior. |
| `shift_hours` | `float = 8.0` | The budget `B = shift_hours * 3600` seconds when `use_time_budget`. |
| `num_vehicles` | `int = 1` | Reps to route (TOP). `1` => identical to today. |
| `vehicle_starts` / `vehicle_ends` | `tuple[int,...] \| None = None` | Per-rep depots; `None` => all share the depot. |
| `distance_engine` | `Literal["haversine","osrm"] = "haversine"` | Selects the travel-time backend. |
| `osrm_url` | `str = "http://localhost:5000"` | OSRM Table service root when `distance_engine="osrm"`. |

With all defaults, `plan_route` produces byte-identical routes to today.

## Files to modify / create

```
src/nba/routing/tsp_profits.py     # add time budget + multi-vehicle params (back-compatible)
src/nba/routing/distance.py        # implement OSRMEngine.time_matrix (was NotImplementedError)
src/nba/pipeline/orchestrator.py   # pass new settings through; choose engine by flag
src/nba/config.py                  # the flags above
tests/test_routing.py              # budget + team assertions
tests/test_distance.py             # OSRM client (mocked HTTP)
```

## `tsp_profits.solve_tsp_profits` — additive parameters

```python
def solve_tsp_profits(
    coords, profits, time_matrix, *,
    depot: int = 0,
    capacity: int | None = None,
    time_windows: list[tuple[int,int]] | None = None,
    service_time_s: float = 120.0,
    drop_scale: float = 1000.0,
    lambda_travel: float = 1.0,
    time_limit_s: float = 5.0,
    seed: int = 7,
    # NEW (all optional, defaults reproduce current behavior):
    route_budget_s: float | None = None,        # OP: bound end-node cumulative Time
    num_vehicles: int = 1,                       # TOP
    starts: list[int] | None = None,             # per-rep start nodes
    ends: list[int] | None = None,               # per-rep end nodes
) -> Route | list[Route]:                        # list when num_vehicles > 1
```

- **Time budget (OP):** when `route_budget_s` is set, after building the `Time` dimension, call
  `time_dim.CumulVar(routing.End(v)).SetMax(int(route_budget_s))` for each vehicle `v`. One line per
  the doc 11 §4.1 recipe; turns the PCTSP into a true OP.
- **Team (TOP):** `RoutingIndexManager(n, num_vehicles, depot)` when reps share a depot, or the
  `(n, num_vehicles, starts, ends)` overload otherwise. Each non-depot node keeps its single
  `AddDisjunction`, so a door is served by at most one rep (no double-serving). Decode one `Route`
  per vehicle.
- **Back-compat:** `num_vehicles == 1 and route_budget_s is None` takes the *exact* existing code
  path and returns a single `Route` (not a list), so all Phase 6 tests pass verbatim.

## `distance.OSRMEngine.time_matrix` — implement the stub

```python
def time_matrix(self, coords) -> np.ndarray:
    # GET {base_url}/table/v1/foot/{lon,lat;...}?annotations=duration
    # parse response["durations"] -> (n,n) float64 seconds; symmetrize; zero diagonal
    # raise RoutingError on non-200 / malformed; offline-first: callers may catch & fall back
```

- Conforms to the existing `DistanceEngine` protocol, so **no caller changes** — the orchestrator
  just constructs `OSRMEngine` instead of `HaversineEngine` when `distance_engine="osrm"`.
- Network access is opt-in (flag off by default); tests mock the HTTP call so CI stays offline.

## `orchestrator.plan_route` — wire the flags through

- Build the engine from `settings.distance_engine` (Haversine default).
- Pass `route_budget_s = settings.shift_hours*3600 if settings.use_time_budget else None`,
  `num_vehicles=settings.num_vehicles`, and starts/ends.
- When `num_vehicles > 1`, `plan_route` returns the per-rep routes; `replan` re-solves over the
  union of not-yet-visited doors. Keep the single-`Route` return when `num_vehicles == 1`.

## Tests

`tests/test_routing.py`
- **Budget honored:** with `route_budget_s = B`, every returned route's `total_time_s <= B`; lowering
  `B` drops more doors.
- **Team, no double-serve:** with `num_vehicles=k`, the union of `visited` across routes has no
  duplicates and equals the served set; each route is independently walkable.
- **Back-compat:** `num_vehicles=1, route_budget_s=None` returns a single `Route` equal to the Phase 6
  result for the same instance/seed.

`tests/test_distance.py`
- `OSRMEngine.time_matrix` parses a mocked `durations` block into a symmetric, zero-diagonal matrix;
  raises on a non-200 response; still `isinstance(DistanceEngine)`.

## Leaderboard entry (lift/regression)

Records into the [Phase 17 leaderboard](phase-17-experiment-leaderboard.md): one row per sub-upgrade —
`phase10-budget` (`NBA_USE_TIME_BUDGET=1`), `phase10-team2` (`NBA_NUM_VEHICLES=2`), `phase10-osrm`
(`NBA_DISTANCE_ENGINE=osrm`). Primary metric: **realized shift value** (team throughput / budget
realism should lift it), with `route_time_s_mean` and feasibility as sanity. Expected verdict
**lift** for budget/team; OSRM is judged on realism (may be **neutral** on simulated coords). Each
must clear the DR gate to count as a lift.

## Acceptance

- A budgeted route never exceeds `shift_hours`; a team route partitions doors across reps without
  double-serving; swapping `distance_engine="osrm"` changes only the matrix, not callers.
- All defaults reproduce today's single-vehicle, window-only routes deterministically.
- `ruff` / `pyright` clean; `pytest` green.
