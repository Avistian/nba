# Phase 6 — Routing / TSP-with-Profits

**Depends on:** Phase 1 (schema). **Parallelizable with:** Phases 3–5. **Goal:** turn per-door
profits into a *walkable* route. A Haversine time matrix behind a `DistanceEngine` interface,
KMeans walkable territories, and an OR-Tools **TSP-with-Profits** solver that may *skip* low-value
far-flung doors via drop penalties tied to each door's profit, under time-window and capacity
constraints. "The router disposes."

## Files to create

```
src/nba/routing/__init__.py
src/nba/routing/distance.py
src/nba/routing/territories.py
src/nba/routing/tsp_profits.py
tests/test_distance.py
tests/test_territories.py
tests/test_routing.py
```

## `src/nba/routing/distance.py`

```python
@runtime_checkable
class DistanceEngine(Protocol):
    def time_matrix(self, coords: Sequence[tuple[float, float]]) -> np.ndarray: ...
    # returns (n, n) travel TIME in seconds; symmetric, zero diagonal

class HaversineEngine:
    def __init__(self, *, speed_kmh: float): ...
    def _haversine_km(self, a, b) -> float:                 # great-circle distance
    def time_matrix(self, coords) -> np.ndarray:            # km / speed → seconds; vectorized
        # use broadcasting over lat/lon arrays; assert symmetric, diag 0

class OSRMEngine:
    """Stub conforming to DistanceEngine for a future real road network.
       __init__(base_url); time_matrix → calls OSRM /table; raises NotImplementedError for now,
       with a documented request/response contract so it can drop in without touching callers."""
```

- Distances are **time**, not raw km, so time windows and walking speed compose naturally.
- Haversine is vectorized (no Python double loop) — matrices can be hundreds of doors.

## `src/nba/routing/territories.py`

```python
@dataclass(frozen=True)
class Territory:
    label: int
    indices: list[int]                # indices into the original door list
    centroid: tuple[float, float]

def cluster_territories(coords, *, target_size: int, seed: int) -> list[Territory]:
    # n_clusters = ceil(len(coords) / target_size); KMeans(random_state=seed) on (lat,lon)
    # (lat/lon scaled so 1° lat ≈ 1° lon·cos(meanlat) before clustering → roughly equal-area)
def assign_doors(coords, territories) -> dict[int, int]:    # door_index → territory label
```

- Pre-partitioning keeps each TSP-P instance small (OR-Tools scales poorly past a few hundred
  nodes) and keeps a rep within one walkable cluster per route.

## `src/nba/routing/tsp_profits.py`

```python
@dataclass(frozen=True)
class Route:
    order: list[int]          # visiting sequence incl. depot at ends (original indices)
    visited: list[int]        # doors actually serviced (excludes depot)
    dropped: list[int]        # doors skipped (paid disjunction penalty)
    total_time_s: float
    total_profit: float

def solve_tsp_profits(
    coords, profits, time_matrix, *,
    depot: int = 0,
    capacity: int | None = None,         # max doors serviced (shift_capacity)
    time_windows: list[tuple[int,int]] | None = None,   # per-node seconds-of-day open/close
    service_time_s: float = 120.0,       # dwell per door (knock + pitch)
    drop_scale: float = 1000.0,          # profit→integer penalty scale
    lambda_travel: float = 1.0,          # weight on travel time in arc cost
    time_limit_s: float = 5.0,
    seed: int = 7,
) -> Route:
    # OR-Tools RoutingIndexManager(n, 1 vehicle, depot) + RoutingModel
    # arc cost callback = int(lambda_travel * time_matrix[i,j])
    # AddDisjunction([node], penalty=int(profit[node]*drop_scale)) for each non-depot node
    #   → skipping a high-profit door is expensive; low-profit far door gets dropped
    # Capacity dimension: demand 1 per node, vehicle cap = capacity (if set)
    # Time dimension: transit = travel + service_time; per-node CumulVar.SetRange(open, close)
    # SearchParameters: PATH_CHEAPEST_ARC first solution + GUIDED_LOCAL_SEARCH metaheuristic,
    #   FromSeconds(time_limit_s), log_search=False
    # decode solution → Route; if infeasible, raise RoutingError with diagnostics
```

- **Core idea (TSP-with-Profits):** classic TSP visits everything; here every door is *optional*
  with a drop penalty equal to its (scaled) profit. The solver trades travel time against profit
  collected, so geographically isolated low-reward doors are dropped — exactly the field behavior
  we want. Drop penalty integerization via `drop_scale` (OR-Tools needs int costs).
- **Constraints:** capacity caps doors per shift; time windows model "residential 16:00–19:00";
  service time models dwell. All optional so tests can isolate each.
- Determinism: fixed `seed`, fixed `time_limit_s`, single thread → reproducible routes.

## Tests

`tests/test_distance.py`
- `HaversineEngine.time_matrix` is symmetric, zero diagonal, non-negative; known-distance check
  (e.g. two points ~1 km apart → time ≈ 1 km / speed).
- `OSRMEngine` conforms to `DistanceEngine` (isinstance) and raises `NotImplementedError`.

`tests/test_territories.py`
- `cluster_territories` produces `ceil(n/target_size)` clusters; every door assigned exactly once;
  deterministic by seed; nearby doors share a label more often than far ones.

`tests/test_routing.py`
- **Drops outliers:** dense high-profit cluster + one far low-profit door → far door in `dropped`.
- **Keeps profit:** raising a door's profit above its travel cost moves it from `dropped`→`visited`.
- **Capacity:** `capacity=k` → `len(visited) ≤ k`.
- **Time windows:** a door with an impossible window is dropped or ordered within window; cumulative
  arrival times respect ranges.
- **Determinism:** identical inputs+seed → identical `Route`.
- **Degenerate:** single door (besides depot) → trivially visited or dropped by its penalty;
  empty door set → empty route, no crash.

## Acceptance

- `solve_tsp_profits` returns a walkable route that visits dense high-profit doors and drops
  far-flung low-profit ones, respecting capacity and time windows, deterministically.
- `DistanceEngine` is swappable (Haversine now, OSRM stub validates the seam).
- `ruff`/`pyright` clean; `pytest` green.
