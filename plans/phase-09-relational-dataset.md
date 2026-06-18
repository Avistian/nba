# Phase 9 — Relational dataset (new, mirrors the flat one)

**Depends on:** Phase 1 (schema), Phase 2 (flat simulator). **Parallelizable with:** Phases 10-13
(those routing/learning upgrades run on either dataset). **Goal:** add a **second, relational
dataset** that *mirrors* the existing flat simulator — same `BanditEvent` stream, same offline-first
determinism, same oracle isolation — but whose ground truth carries **genuine relational and
temporal structure** (households, neighbor edges, per-prospect interaction histories, competitor
overlap) and which can **emit a heterogeneous graph** for a future Relational Deep Learning value
model ([Phase 14](phase-14-relational-deep-learning.md)). The flat dataset stays the default; the
relational one is selected by a flag. Grounded in
[docs/12 §2, §6](../docs/12-relational-deep-learning-mixin.md) and the step-by-step build in
[docs/13](../docs/13-relational-dataset.md).

> **Why first.** Doc 12 §5.2 is explicit: today's simulator samples *independent* prospects, so a
> GNN has no graph to learn from. Building relational structure is the prerequisite for RDL — and it
> also gives the dynamic/stochastic and decision-focused phases richer signal to exploit. It changes
> **no existing behavior**: with `dataset_mode="flat"` (the default) the new code path is never taken.

## Feature flags (added to `src/nba/config.py` `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `dataset_mode` | `Literal["flat","relational"] = "flat"` | Selects which simulator the scripts/demo use. Default reproduces today exactly. |
| `relational_data_dir` | `Path = Path("data/relational")` | Where relational logs + entity tables + graph land (separate from `data/`). |
| `n_households` | `int = 0` | `0` => derive from `n` (≈ 1 household per 3 doors). Groups addresses into household/account nodes. |
| `neighbor_radius_km` | `float = 0.15` | Two doors get a `near` edge if within this great-circle radius. |
| `history_len` | `int = 8` | Max prior interactions retained per prospect in the temporal history. |
| `competitor_density` | `float = 0.2` | Fraction of blocks carrying a competitor-overlap signal/edge. |
| `relational_seed` | `int = 7` | Independent seed so the relational world is reproducible. |

**Default-off guarantee:** all consumers read `dataset_mode`; with the default `"flat"` none of the
new modules are imported on the hot path, and the existing 143 tests are unaffected.

## Files to create

```
src/nba/data/relational_simulator.py   # relational ground-truth world + logging policy
src/nba/data/graph.py                   # heterogeneous-graph builder + graph allow-list
scripts/generate_relational_logs.py     # CLI: relational logs + entity tables + graph artifacts
tests/test_relational_simulator.py
tests/test_graph.py
```

No existing file is modified except `config.py` (new flags) and — additively — the scripts that
already branch on settings.

## `src/nba/data/relational_simulator.py`

Mirrors the public surface of `data/simulator.py` so it is a drop-in *alternative*, not a rewrite.
The flat oracle in `simulator.py` is left untouched.

### Entity model (the relational substrate)

```python
@dataclass(frozen=True)
class Household:          # account/household node
    household_id: str
    address_ids: list[str]
    centroid: tuple[float, float]

@dataclass(frozen=True)
class Interaction:        # one historical touch (temporal edge customer<->time)
    address_id: str
    action: Action
    outcome: Outcome
    ts: datetime

@dataclass(frozen=True)
class RelationalWorld:    # the sampled population + relationships
    contexts: dict[str, ProspectContext]      # by address_id
    households: list[Household]
    near_edges: list[tuple[str, str]]          # spatial proximity (<= neighbor_radius_km)
    household_edges: list[tuple[str, str]]     # same-household co-membership
    competitor_edges: list[tuple[str, str]]    # shared-competitor overlap
    histories: dict[str, list[Interaction]]    # per-prospect interaction history
```

### Sampling with structure (vs. flat's i.i.d. draws)

```python
def sample_world(n: int, *, settings: Settings, seed: int) -> RelationalWorld:
    """Draw n Ames-backed doors, group them into households, build near/household/competitor
    edges, and synthesize a per-prospect interaction history. Reuses ames.load_ames and
    simulator.sample_context for the per-door block so non-relational fields match the flat world."""
```

- **Households:** spatially-coherent clusters (reuse `routing.territories`-style grouping on lat/lon)
  collapsed to a household node; `prior_interactions` becomes *consistent* with the door's history.
- **Neighbor edges:** vectorized pairwise haversine (reuse `routing.distance`) thresholded at
  `neighbor_radius_km`.
- **History:** each prospect gets up to `history_len` timestamped `Interaction`s drawn from the
  flat behavior policy, so `prior_interactions` and `neighbor_recent_conversion` are now *explained*
  by an actual event log rather than sampled in isolation.

### Latent ground truth (relational + temporal effects) — ORACLE

```python
def latent_scores(ctx, action, *, world: RelationalWorld) -> dict[Outcome, float]:
    """Extends simulator.latent_scores with effects that REQUIRE the graph:
       - neighbor social proof: nearby recent CLOSED/APPOINTMENT lift this door's APPOINTMENT/CLOSED
       - household momentum: a prior CLOSED in the same household lifts engagement
       - interaction fatigue: realistic decay over the door's own history (temporal)
       - competitor overlap: a shared-competitor edge depresses CLOSED
    Falls back to the flat effects when an edge set is empty, so a degenerate world == flat world."""
def true_reward(ctx, action, *, world) -> float
def true_best_action(ctx, *, world) -> Action
```

- **Oracle isolation (unchanged rule):** `latent_scores`/`true_reward`/`true_best_action` here, like
  in the flat simulator, must never be imported by `nba.reward`, `nba.bandits`, `nba.ope`,
  `nba.routing`, `nba.pipeline`, or `nba.api`. The Phase 2 AST guard
  (`tests/test_ethics.py::test_no_oracle_leak`) is **extended** to scan this module too.

### Logging policy + event generation (identical contract to flat)

```python
def behavior_policy(ctx, rng, *, world, temp=0.5) -> tuple[Action, float]   # full support, p>0
def generate_logs(n, *, settings, seed) -> tuple[list[BanditEvent], RelationalWorld]
def logs_to_frame(events) -> pd.DataFrame    # same columns as flat + an optional household_id col
```

The emitted `BanditEvent`s are **schema-identical** to the flat ones, so `RewardModel`, the bandits,
OPE, the orchestrator and the API consume them unchanged. The extra relational structure is carried
*alongside* in the `RelationalWorld` / graph artifacts, consumed only by the RDL model.

## `src/nba/data/graph.py`

Turns a `RelationalWorld` into a typed graph for the RDL encoder, behind a **graph allow-list** that
mirrors `features.ALLOWED_FEATURES` (doc 12 §5.3).

```python
GRAPH_NODE_FEATURES: dict[str, tuple[str, ...]]   # per node type, the allow-listed fields
GRAPH_EDGE_TYPES: tuple[str, ...] = ("near", "same_household", "shares_competitor", "interacted")

@dataclass(frozen=True)
class HeteroGraph:
    node_features: dict[str, np.ndarray]          # type -> (n_nodes, n_feats)
    edge_index: dict[str, np.ndarray]             # edge_type -> (2, n_edges)
    node_ids: dict[str, list[str]]

def build_graph(world: RelationalWorld) -> HeteroGraph:
    """Assemble nodes/edges using ONLY GRAPH_NODE_FEATURES; lat/lon/address_id and any protected
    field are excluded by construction, exactly like the flat featurizer."""
def allowed_node_feature(node_type: str, field: str) -> bool
```

- **No torch dependency here.** `build_graph` returns plain numpy/index arrays; the optional
  `torch_geometric` conversion lives in [Phase 14](phase-14-relational-deep-learning.md) so this
  phase adds **zero heavy deps**.
- **Ethics at the graph layer:** the allow-list is enforced when nodes/edges are built, so a forbidden
  attribute can enter neither a node feature **nor** propagate via an edge (the doc 12 §5.3 leakage
  concern). Tested in `test_graph.py`.

## `scripts/generate_relational_logs.py`

- CLI (`argparse`): `--n 20000 --seed 7 --out data/relational`.
- Writes: `logs.parquet` (BanditEvents), `households.parquet`, `edges.parquet`, and a serialized
  `graph.npz`. Prints arm frequencies, min propensity (> 0), node/edge counts, and the fraction of
  doors with a non-empty neighborhood (sanity that the world is actually relational).

## Tests

`tests/test_relational_simulator.py`
- **Mirror contract:** emitted `BanditEvent`s validate against the schema and round-trip through
  `logs_to_frame`/`frame_to_events` identically to the flat simulator.
- **Positivity / overlap:** every `propensity > 0`; all 5 arms represented across many contexts.
- **Reproducibility:** `generate_logs(n, seed=7)` twice => identical frames and identical world.
- **Relational signal is real:** holding a door's own features fixed, adding a nearby recent `CLOSED`
  raises `true_reward(KNOCK_NOW)` (social proof); a shared-competitor edge lowers it. (Oracle-only
  assertions, never in a learner.)
- **Degenerate == flat:** with no edges/history, `true_reward` matches `simulator.true_reward` within
  tolerance, proving the relational world is a strict superset.

`tests/test_graph.py`
- `build_graph` yields the declared node/edge types; counts match the world.
- **Allow-list:** `lat`, `lon`, `address_id`, and any protected probe field never appear in any
  `node_features` block; an edge cannot carry a forbidden attribute.
- Deterministic by seed.

`tests/test_ethics.py` (extended)
- `test_no_oracle_leak` also scans `nba.data.relational_simulator` so its oracle symbols cannot be
  imported by learning modules.

## Leaderboard entry (lift/regression)

Records into the [Phase 17 leaderboard](phase-17-experiment-leaderboard.md): run
`scripts/run_experiment.py --experiment-id phase09-relational --phase 09 --dataset relational` with
the **unchanged** LightGBM model and router. The dataset is a *substrate*, not a value change, so the
expected verdict is **neutral** — the test is that swapping `dataset_mode` alone does **not regress**
realized shift value vs `baseline`. The actual lift on relational data is claimed later by
[Phase 14](phase-14-relational-deep-learning.md). A regression here means the relational world is
mis-calibrated against the flat one and must be fixed before RDL.

## Acceptance — ✅ built

- [x] `python scripts/generate_relational_logs.py --n 5000 --seed 7` writes logs + entity tables + a
  graph, min propensity > 0, all arms present, and a non-trivial edge count.
- [x] With `dataset_mode="flat"` (default) nothing changes: existing scripts, demo, and tests behave
  identically (no new heavy deps imported) — verified by `tests/test_demo_dataset_modes.py`.
- [x] Oracle isolation holds for the new module (`tests/test_ethics.py::test_no_oracle_leak` extended);
  the graph allow-list blocks geo/identity/protected fields at construction (`tests/test_graph.py`).
- [x] `ruff` / `pyright` clean; `pytest` green.
- Built via `src/nba/data/relational_simulator.py`, `src/nba/data/graph.py`,
  `scripts/generate_relational_logs.py`. Data-contract decision:
  [decisions/2026-06-18-relational-dataset-contract.md](../decisions/2026-06-18-relational-dataset-contract.md)
  (sidecars + optional `household_id` column; `BanditEvent` unchanged).
