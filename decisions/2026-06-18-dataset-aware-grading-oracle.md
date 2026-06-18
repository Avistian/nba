# Decision: Grade experiments through a dataset-aware oracle facade, not direct simulator calls

## Decision

`scripts/run_demo.py` and the Phase 17 leaderboard reach ground truth only through a small facade,
`src/nba/eval/oracle.py`: an `Oracle` Protocol (`true_reward`, `true_best_action`, `sample_outcome`)
with a `FlatOracle` (a pure pass-through to `nba.data.simulator`) and a `RelationalOracle` (bound to a
sampled `RelationalWorld`), selected by `oracle_for(settings, *, world=None)` on
`Settings.dataset_mode`. The flat path forwards to the identical functions, so flat output stays
byte-identical (proven by a `seed=7` determinism regression).

## Context

Phase 17 must grade flag configs on **either** the flat or the relational dataset. The demo previously
imported `true_reward`/`true_best_action`/`sample_outcome` directly from the flat simulator. The
relational oracle additionally needs the `RelationalWorld` (its social-proof reward depends on
`near_edges`/geography), which a direct import can't thread cleanly.

## Alternatives considered

- **Keep direct simulator imports and branch inline in `run_demo`.** Rejected: scatters
  `dataset_mode` conditionals through the script and re-imports oracle symbols in two shapes, making
  the no-leak guard and the flat-determinism guarantee harder to keep.
- **Make the relational oracle stateless (recompute the world per call).** Rejected: re-deriving
  `near_edges` on every reward call is wasteful and risks drift when the demo repositions doors onto a
  dense block — the bound world is rebuilt once via `world_from_contexts` instead.

## Reasoning

A narrow protocol with two implementations keeps grading dataset-aware behind a stable seam, leaves
every learner untouched, and confines all oracle access to one module that `nba.eval` (already
excluded from the AST guard) owns. The relational reward stays consistent when geography changes
because the oracle carries the world.

## Trade-offs accepted

- One extra indirection layer between the demo and the simulator.
- `run_demo` must rebuild the relational world's `near_edges` when it repositions doors, an explicit
  step the flat path doesn't need (flat repositioning is reward-neutral).

## Supersedes

None.
