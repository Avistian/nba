# Dataset & Evaluation — Rules

Confirmed patterns — apply by default on dataset (Phase 9) and leaderboard (Phase 17) work.

## R1: New datasets mirror the flat `BanditEvent` contract

A new dataset (e.g. `dataset_mode="relational"`) must emit a **schema-identical** `BanditEvent`
stream. Do not widen `BanditEvent`/`ProspectContext`. Carry extra structure in additive sidecar
artifacts (`households.parquet`, `edges.parquet`, `graph.npz`) and, at most, optional non-model
DataFrame columns (e.g. `household_id`) that `frame_to_events` ignores. `ProspectContext` keeps
`extra="forbid"`.

*Confirmed: Phase 9, `tests/test_relational_simulator.py` (round-trip), decision
2026-06-18-relational-dataset-contract.*

## R2: A new world must degenerate to the flat oracle

A relational/temporal world with no edges and no history must reproduce the flat `true_reward` within
tolerance. The new world is a strict superset of the old one, not a different calibration.

*Confirmed: `tests/test_relational_simulator.py::degenerate==flat`.*

## R3: Oracle access for grading goes through `eval/oracle.py`

Scripts/eval reach `true_reward`/`true_best_action`/`sample_outcome` only via `oracle_for(settings,
world=...)`. `FlatOracle` is a pure pass-through (flat numbers byte-identical); `RelationalOracle`
binds the sampled world. Never import oracle symbols directly into a learner. The relational simulator
oracle is also covered by the no-leak AST guard.

*Confirmed: Phase 17, `tests/test_demo_dataset_modes.py`, `tests/test_ethics.py`.*

## R4: The graph node/edge allow-list mirrors `features.ALLOWED_FEATURES`

`graph.build_graph` may only emit node features on the graph allow-list; geo/identity/protected fields
are excluded by construction (a GNN makes leakage *easier* via message passing). Verify with a test.

*Confirmed: `tests/test_graph.py`.*

## R5: The leaderboard is append-only; a lift needs gate + primary gain

`leaderboard.jsonl` is append-only (mirrors the event store). A run is a **lift** only if the primary
metric (realized shift value) improves **and** the DR lower bound clears baseline + `ope_min_lift`; a
primary drop beyond `ope_min_lift` is a **regression**; otherwise **neutral**. Deltas are
sign-normalized so `+` always means "better".

*Confirmed: Phase 17, `tests/test_leaderboard.py`.*

## R6: A new dataset substrate is graded "neutral", not "lift"

Swapping `dataset_mode` with an unchanged model/router must not regress vs baseline; the expected
verdict is neutral. The value lift on relational data is claimed later by the RDL model (Phase 14),
not by the dataset itself.

*Confirmed: shipped `phase09-relational` row is neutral.*
