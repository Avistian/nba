## Decision: Append-only SQLite event store for serving logs

## Context

The online loop must persist every decision (with propensity) and later outcomes for retraining and
OPE. Phase 7 needed a queryable store without building full data infrastructure.

## Alternatives considered

1. **Parquet append-only files** — simple, batch-friendly, awkward for concurrent single-row writes.
2. **SQLite append-only** — local, queryable, WAL mode, no UPDATE/DELETE on decisions.
3. **Managed DB (Postgres)** — production-grade, deferred for prototype scope.

## Reasoning

SQLite matches local prototype + thin FastAPI scope. Append-only schema preserves audit trail and
matches OPE's need for immutable decision records. Parquet remains for batch simulator logs.

## Trade-offs accepted

- Single-writer concurrency limits; not multi-rep production scale yet.
- Corrections require new outcome rows (slightly more storage).

## Supersedes

(none)
