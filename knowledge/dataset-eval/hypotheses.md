# Dataset & Evaluation — Hypotheses

Need more data before promoting to rules.

## H1: An RDL value model will beat LightGBM on relational data through the same DR gate

The relational dataset adds graph-dependent effects (social proof, household momentum, competitor
overlap) that hand-crafted flat features only summarize. Hypothesis: a calibrated GNN `q(x,a)` over the
Phase 9 graph will record a **lift** vs the flat LightGBM baseline on relational logs.

*Evidence so far: 0 — Phase 14 not built. The `phase09-relational` row is deliberately neutral
(dataset substrate only). Test when the RDL model lands.*

## H2: The leaderboard's default `eval_n_shifts`/`eval_seeds` give stable verdicts

`evaluate` aggregates over `eval_n_shifts` shifts × `eval_seeds`. Hypothesis: the current defaults keep
the primary-metric delta's noise below `ope_min_lift`, so a true neutral never flips to lift/regression
across re-runs.

*Evidence: shipped baseline + relational rows reproduce by seed; needs repeated runs at varied
n_shifts to confirm verdict stability.*

## H3: Sidecar artifacts (graph.npz) stay small enough to commit/regenerate cheaply

The relational graph is plain numpy `npz`. Hypothesis: at production log scale the graph + entity tables
remain cheap to regenerate per run, so they need not be versioned as large binaries.

*Evidence: 1 generation at n≈20k. Needs scale tests before relying on regenerate-on-demand.*
