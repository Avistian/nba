# Phase 14 — Relational Deep Learning value model (GNN behind the QModel protocol)

**Depends on:** Phase 9 (relational dataset + graph), Phase 3 (reward model + calibration), Phase 5
(OPE gate). **Goal:** add an **optional, evidence-gated** value model that learns directly from the
relational graph — a GNN (R-GCN / GraphSAGE) producing `q(x, a)` — as a drop-in alternative to
LightGBM behind the `QModel` protocol. It replaces (or augments) exactly **one box**: the reward
model. Everything else (bandit, OPE, router, ethics) is untouched. Grounded in
[docs/12](../docs/12-relational-deep-learning-mixin.md) and the build in
[docs/18](../docs/18-relational-deep-learning.md).

> **Contained, reversible experiment** (doc 12 §3): the orchestrator talks to the reward model only
> through `QModel.q_all`. A GNN implementing that protocol drops in without touching the bandit, OPE,
> router, API, or ethics code. Adopt it **only if** it beats LightGBM through the same DR gate on
> **route value** (doc 12 §6, §9). If it ties, keep LightGBM — it's cheaper.

## Feature flags (added to `src/nba/config.py` `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `reward_model_kind` | `Literal["lightgbm","rdl"] = "lightgbm"` | Which value model the scripts/orchestrator build. Default = today. |
| `gnn_type` | `Literal["rgcn","graphsage"] = "graphsage"` | Message-passing layer family. |
| `gnn_layers` | `int = 2` | Message-passing rounds (neighbor hops). |
| `gnn_hidden` | `int = 64` | Hidden embedding width. |
| `gnn_epochs` | `int = 50` | Training epochs. |
| `gnn_lr` | `float = 1e-3` | Learning rate. |

**Heavy deps are opt-in.** `torch` / `torch_geometric` are an **optional dependency group**
(`pip install nba[rdl]` / `uv sync --extra rdl`). With `reward_model_kind="lightgbm"` they are never
imported, so the base install and the existing tests are unchanged.

## Files to create / modify

```
src/nba/reward/graph_model.py    # GraphRewardModel(QModel): build PyG graph, R-GCN/GraphSAGE, q head
src/nba/reward/calibrate.py      # shared isotonic wrapper reused by LightGBM + GNN (refactor, optional)
src/nba/config.py                # the flags above + optional-extra wiring
pyproject.toml                   # [project.optional-dependencies] rdl = ["torch", "torch_geometric"]
scripts/train_reward.py          # branch on reward_model_kind (additive)
tests/test_graph_model.py
```

## `reward/graph_model.py`

```python
class GraphRewardModel:               # implements QModel (and QEnsemble via MC-dropout, optional)
    @classmethod
    def fit(cls, events, world, *, settings) -> "GraphRewardModel":
        """Build the heterogeneous graph (data.graph.build_graph), run gnn_layers of message
        passing, attach a q(x,a) head, train on logged (context, action, reward) tuples, then fit
        an isotonic calibrator on a held-out split (same discipline as RewardModel)."""
    def q(self, ctx, action) -> float
    def q_all(self, ctx, actions=ACTIONS) -> np.ndarray   # one inference subgraph per context
    def best_action(self, ctx, actions=ACTIONS) -> Action
    def save(self, model_dir) / load(model_dir)           # state_dict + calibrator + feature manifest
```

### The non-negotiable rails (doc 12 §5.3)

- **Ethics allow-list at the graph layer.** Features enter only via `data.graph.GRAPH_NODE_FEATURES`;
  a node cannot absorb a forbidden attribute from a neighbor, and proximity cannot encode a protected
  identity. A test mirrors `tests/test_ethics.py`: forbidden probe fields can enter neither nodes
  **nor** edges, and message passing cannot recover them.
- **Calibration.** The GNN output is wrapped in the *same* isotonic calibrator as LightGBM, because
  DM/DR OPE average `q̂` directly and neural nets are famously miscalibrated (doc 12 §5.3).
- **No oracle leak.** Trains only on logged tuples; the AST guard covers the new module.
- **Frozen schema.** A graph-feature manifest is persisted and verified on load, like
  `feature_names.json`, preventing train/serve skew.

## `scripts/train_reward.py` (additive branch)

- `reward_model_kind="lightgbm"` => unchanged path.
- `reward_model_kind="rdl"` => load the relational logs + graph (Phase 9), fit `GraphRewardModel`,
  calibrate, save, and emit `metrics.json` with prediction error **and** a head-to-head route-value /
  OPE comparison vs. the LightGBM baseline (doc 12 §6.4).

## Tests

`tests/test_graph_model.py` (gated on the `rdl` extra; skipped if torch absent)
- **QModel conformance:** `GraphRewardModel` satisfies the protocol; `q_all` returns `|ACTIONS|`
  scores; save/load round-trips with the schema guard.
- **Graph allow-list / leakage:** forbidden fields never enter node or edge features; a planted
  protected signal is not recoverable from embeddings (statistical probe test).
- **Calibration present:** predictions pass the same calibration sanity checks as LightGBM.
- **Bench harness exists:** a (small, seeded) comparison computes route value + DR for both models so
  promotion uses the same gate.

## Leaderboard entry (lift/regression)

Records into the [Phase 17 leaderboard](phase-17-experiment-leaderboard.md): `phase14-rdl`
(`NBA_REWARD_MODEL_KIND=rdl --dataset relational`). It is compared not only to `baseline` but
**head-to-head against `phase09-relational`** (LightGBM on the same relational data), so the row
isolates the GNN's contribution from the dataset's. Judged on the **primary metric** + `ope_lcb`.
Expected verdict **lift only if it beats LightGBM-on-relational through the DR gate**; otherwise the
honest verdict is **neutral** and LightGBM stays (doc 12 §9). This row is the entire justification for
adopting RDL.

## Acceptance

- `GraphRewardModel` is a true drop-in: swapping `reward_model_kind="rdl"` changes only the model, not
  the bandit/OPE/router/API.
- It promotes **only** if it clears the existing DR gate on route value (doc 12 §9); otherwise
  LightGBM stays. Calibration + ethics allow-list + oracle isolation all hold.
- Base install (no `rdl` extra) is unchanged; `ruff` / `pyright` clean; `pytest` green (RDL tests
  skipped without torch).
