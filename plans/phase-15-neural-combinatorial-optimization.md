# Phase 15 — Upgrade 4: Neural combinatorial optimization (DEFERRED / optional)

**Status: deferred.** Adopt **only** when you have a measured reason to (doc 11 §7). OR-Tools solves
neighborhood-sized instances to near-optimality in seconds, deterministically, with first-class time
windows and capacities. A learned router trades determinism and optimality *guarantees* for **speed
at scale** and amortized solving across similar instances. This phase is specified so the seam is
understood, but it should not be built speculatively.

**Depends on:** Phase 10 (orienteering objective), optionally Phase 14 (GNN encoder). **Goal:** an
attention-based encoder-decoder that emits routes directly, behind a `Router` protocol, with OR-Tools
retained as the **reference oracle** in tests. Grounded in
[docs/11 §7](../docs/11-improving-nba-spatio-relational-optimization.md) and
[docs/19](../docs/19-neural-combinatorial-optimization.md).

## When to build it (the gate, doc 11 §7.2)

- You must (re)route **thousands** of doors **many times per minute** (a large live fleet), **or**
- you want the router to exploit statistical regularities across your specific neighborhoods, **or**
- you're pursuing the end-to-end vision where value model + router are one network ([Phase 16](phase-16-decision-focused-rdl.md)).

If none hold, **do not build this** — keep OR-Tools.

## Feature flags (when built)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `router_kind` | `Literal["ortools","neural"] = "ortools"` | Selects the router. Default = today's exact solver. |
| `neural_router_ckpt` | `Path \| None = None` | Trained policy checkpoint. |
| `optimality_gap_tol` | `float = 0.05` | Max allowed value gap vs. OR-Tools in tests. |

## Sketch

```
src/nba/routing/base.py            # Router protocol: solve(coords, profits, time_matrix, ...) -> Route
src/nba/routing/neural_router.py   # attention encoder-decoder (Kool et al. 2019), REINFORCE-trained
src/nba/routing/tsp_profits.py     # wrap existing solver as the OR-Tools Router implementation
scripts/train_router.py
tests/test_neural_router.py
```

- **Architecture:** attention encoder over node features (coords + prize, optionally the Phase 14 GNN
  embeddings as the encoder front end), autoregressive pointer decoder, trained by REINFORCE / POMO to
  maximize route value (doc 11 §7.1).
- **Honest caveat (doc 11 §7.3):** a learned router is **heuristic** — never advertise "guaranteed
  optimal." Keep OR-Tools as the test oracle and assert the learned router stays within
  `optimality_gap_tol` on held-out instances.

## Leaderboard entry (lift/regression)

Records into the [Phase 17 leaderboard](phase-17-experiment-leaderboard.md): `phase15-neural`
(`NBA_ROUTER_KIND=neural`). Judged on **`optimality_gap`** (must stay within `optimality_gap_tol` of
OR-Tools) and on throughput/latency at scale — **not** on a mean-value lift, since a learned router is
heuristic. Expected verdict **neutral on value, justified by speed**; a value drop beyond the gap
tolerance is a **regression**.

## Acceptance (when built)

- `router_kind="ortools"` (default) is unchanged; `"neural"` stays within the optimality-gap bound of
  OR-Tools on held-out instances; both satisfy the `Router` protocol.
- `ruff` / `pyright` clean; `pytest` green (neural tests skipped without the optional extra).
