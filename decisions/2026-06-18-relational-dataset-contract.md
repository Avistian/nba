# Decision: Keep the BanditEvent data contract unchanged for the relational dataset

## Decision

The relational dataset (Phase 9) does **not** change the `BanditEvent` / `ProspectContext` data
contract in [src/nba/schema.py](../src/nba/schema.py). The relational simulator emits a
schema-identical `BanditEvent` stream. Relational structure is carried **alongside** the logs in
additive sidecar artifacts under `data/relational/` — `households.parquet`, `edges.parquet`,
`graph.npz` — plus **one optional, non-model `household_id` column** on `logs.parquet`. That
`household_id` is a DataFrame column only, never a Pydantic field: `frame_to_events` reads only
`ctx.*` plus a fixed column set, so it ignores the extra column and the round-trip stays identical,
while `ProspectContext` keeps `extra="forbid"` and never sees it.

## Context

The original ask anticipated that "data will probably change the data contract." A relational world
genuinely has more structure (households, neighbor/competitor edges, timestamped interaction
histories). The naive move is to widen `ProspectContext`/`BanditEvent` to carry it. But that contract
is the substrate of the entire verified 0–8 loop and its test suite, and is consumed unchanged by
`RewardModel`, the bandits, OPE, the orchestrator, the API, `train_reward.py`, and `evaluate_policy.py`.

## Alternatives considered

- **Widen `BanditEvent`/`ProspectContext` with relational fields (edges, household, history).**
  Rejected: forces a schema migration on every consumer and the event store, breaks `extra="forbid"`
  invariants, and risks leaking identity/geo structure into the model feature path.
- **A separate, differently-shaped relational log format.** Rejected: would fork every downstream
  reader and defeat the head-to-head LightGBM-vs-RDL comparison through the same OPE gate.
- **Put `household_id` into `ProspectContext`.** Rejected: it is an identity/grouping key, not a
  model feature; keeping it a DataFrame-only column keeps the allow-list honest.

## Reasoning

Backward compatibility is free if the logged event stream is schema-identical and the relational
extras are strictly opt-in sidecars consumed only by the future RDL model (Phase 14). Any existing
consumer of `logs.parquet` works unchanged on relational logs; there is no migration. This preserves
the "new capability behind a stable interface" discipline used everywhere else in the repo, and keeps
the flat path byte-identical (`dataset_mode="flat"` default).

## Trade-offs accepted

- The rich relational structure lives outside the typed event contract, so consumers that want it
  (only the RDL model) must load the sidecars explicitly rather than reading it off `BanditEvent`.
- A small amount of convention (the `household_id` column, the sidecar file layout) is enforced by
  tests and docs rather than by the Pydantic schema.

## Supersedes

None. Refines [2026-06-18-feature-flagged-relational-upgrades.md](2026-06-18-feature-flagged-relational-upgrades.md)
item 2 (relational dataset as a mirror) with the concrete contract-preservation mechanism.
