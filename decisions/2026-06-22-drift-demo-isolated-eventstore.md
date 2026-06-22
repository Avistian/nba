## Decision: Drift demo uses an isolated artifact tree and never mutates production ops paths

## Context: `simulate_drift_demo` originally redirected only `db_path` to `artifacts/drift_demo/events.db`. `bootstrap_deployed`, `RetrainLoop`, and `_score_drift` still wrote `deployed.json`, saved models, drift reports, and audit rows under the default production paths (`artifacts/models`, `artifacts/monitoring/*`). Earlier, `_persist_events` also unconditionally deleted `settings.db_path` before ingest — wiping the shared API EventStore.

## Alternatives considered:
- Remove `unlink()` only and rely on idempotent `ingest_bandit_events` — still mixes demo data into the production store.
- Fail/warn when production db is non-empty — noisy for intentional local workflows.
- Add a `--force-reset-db` CLI flag — extra surface area for a script that should be safe by default.
- Redirect paths inside `bootstrap_deployed` / `RetrainLoop` — couples production retrain code to demo concerns; redirecting once in `run_drift_demo` via `_demo_settings` keeps the seam at the script boundary.

## Reasoning: Demo ingest and retrain artifacts exist to seed Grafana rollups and illustrate the monitor/retrain loop locally, not to mutate production telemetry or overwrite live model manifests. Redirecting every production default (`db_path`, `model_dir`, `deployed_model_manifest`, `monitoring_report_path`, `retrain_audit_path`) to `artifacts/drift_demo/*` isolates demo data while preserving explicit `Settings` overrides. `ingest_bandit_events` already skips duplicate `decision_id`s, so unlink is only needed for the isolated demo db.

## Trade-offs accepted: Grafana quickstart after the demo must point the exporter at drift-demo artifacts (e.g. `NBA_DB_PATH=artifacts/drift_demo/events.db`, `NBA_DEPLOYED_MODEL_MANIFEST=artifacts/drift_demo/models/deployed.json`) unless the operator explicitly overrides paths to another non-default location.

## Supersedes: none (extends the EventStore-only isolation from the same date)
