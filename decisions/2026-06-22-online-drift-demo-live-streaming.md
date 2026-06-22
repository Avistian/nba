## Decision: Online drift demo streams live-stamped events and alerts via email on significant drift

## Context

Phase 18's `simulate_drift_demo.py` computes all shifts in memory and writes artifacts in one burst.
Operators cannot watch drift signals climb on Grafana in real time. The user requested an online demo
that catches drift during running monitoring, with email notification and gated retrain on significant
drift.

## Alternatives considered

- **Serve-API-driven** (`/recommend` + `/feedback` with a drifting client) — more faithful to
  production but much heavier; requires a running uvicorn process and a client simulator.
- **Batch demo with artificial timestamps** — still writes all reports at once; Grafana sees a
  vertical line, not a live animation.
- **Grafana-only alerting** — no email; relies on Docker and does not demonstrate the ops
  notification path.
- **Email inside `RetrainLoop`** — couples network/SMTP to core retrain logic; harder to test and
  violates the side-channel pattern from Phase 18.

## Reasoning

A **producer-driven** script streams synthetic labeled events into the isolated `artifacts/drift_demo/*`
tree over wall-clock time. Each tick:

1. Generates events (in-distribution pre-onset, drifting after).
2. Re-stamps timestamps to `now` (generated logs use a fixed `_BASE_DATE` that breaks window
   splits and cadence without re-stamping).
3. Ingests into the EventStore.
4. Runs `RetrainLoop.run` (one `DriftReport` per tick).
5. Calls `maybe_alert_drift` (email side-channel, debounced, off by default).

Email is a **notification side-channel**, not a new detector. Significance uses the existing
`RetrainTrigger` + breached-signal count. Retrain-on-detection is already what `RetrainLoop` does;
the DR promotion gate still guards shipping.

## Trade-offs accepted

- Synthetic events, not real API traffic — simpler and sufficient for demonstrating the monitor loop.
- SMTP creds supplied via env at run time only; dry-run print when enabled without creds.
- Post-promote ticks use monitor-only scoring (not re-fitting every tick) to keep the demo fast.
- Exporter must be started with demo-tree env vars (`make metrics-exporter-demo`).

## Supersedes

None. Extends [2026-06-22-drift-demo-isolated-eventstore.md](2026-06-22-drift-demo-isolated-eventstore.md).
