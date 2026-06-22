## Decision: Drift demo uses an isolated EventStore path and never unlinks production logs

## Context: `simulate_drift_demo._persist_events` unconditionally deleted `settings.db_path` before ingest. The default path (`artifacts/events.db`) is shared with the API EventStore, so running the demo could wipe live append-only production logs.

## Alternatives considered:
- Remove `unlink()` only and rely on idempotent `ingest_bandit_events` — still mixes demo data into the production store.
- Fail/warn when production db is non-empty — noisy for intentional local workflows.
- Add a `--force-reset-db` CLI flag — extra surface area for a script that should be safe by default.

## Reasoning: Demo ingest exists to seed Grafana rollups for local observability demos, not to mutate production telemetry. Redirecting the default demo path to `artifacts/drift_demo/events.db` isolates demo data while preserving the ability to reset only that store between runs. `ingest_bandit_events` already skips duplicate `decision_id`s, so unlink is only needed for the isolated demo db.

## Trade-offs accepted: Grafana quickstart after the demo must point the exporter at `artifacts/drift_demo/events.db` (e.g. `NBA_DB_PATH=artifacts/drift_demo/events.db`) unless the operator explicitly overrides `db_path` to another non-default location.

## Supersedes: none
