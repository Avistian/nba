# Phase 19 — Online (live-streaming) drift demo + email alerting

**Depends on:** Phase 18 (drift monitoring, retrain loop, Grafana stack, batch drift demo).
**Built after Phase 18** — it turns the batch narrative into a **wall-clock live stream** so operators
can watch drift signals climb, cross thresholds, receive an email alert, trigger a gated retrain, and
see recovery on a live Grafana dashboard. **Day-to-day usage:**
[docs/24-monitoring-operator-guide.md](../docs/24-monitoring-operator-guide.md).

> **Not a new detector.** Drift is still scored by Phase 18 signals/triggers. Phase 19 adds a
> **producer** that streams events into the EventStore over time, **email notification** on
> significant drift, and orchestration targets for the 3-terminal demo flow.

## Problem statement

`simulate_drift_demo.py` computes every shift in memory and writes all artifacts in one burst. The
exporter and Grafana only ever see the **final** state — you cannot watch drift get caught live.

The online demo:

1. **Warm-up** — bootstrap a deployed model from in-distribution logs.
2. **Stream** — ingest labeled events tick-by-tick (with wall-clock timestamps) into an isolated
   `artifacts/drift_demo/*` tree.
3. **Monitor** — run `RetrainLoop.run` once per tick; append one `DriftReport` + audit row each time.
4. **Alert** — on significant drift (`should_retrain` + breached-signal count), send a debounced
   email (or dry-run print when SMTP creds are absent).
5. **Retrain** — same DR gate as Phase 18; promote or hold.
6. **Recover** — post-promote ticks use monitor-only scoring so Grafana shows signal recovery.

## Architecture

```text
run_online_drift_demo.py (tick loop, sleeps)
    -> EventStore (artifacts/drift_demo/events.db)
    -> RetrainLoop.run per tick
    -> drift_reports.jsonl / retrain_audit.jsonl / deployed.json
    -> maybe_alert_drift (email side-channel)
run_metrics_exporter.py (:9091/metrics, demo env vars)
    -> Prometheus (scrape 15s)
    -> Grafana "NBA Ops" (auto-refresh 10s)
```

## Key technical decisions

| Decision | Rationale |
|----------|-----------|
| Isolated `artifacts/drift_demo/*` tree | Per ADR; never mutates production telemetry |
| Re-stamp events to wall-clock `now` per tick | Generated logs use fixed `_BASE_DATE`; without re-stamp, window splits and cadence break |
| One `RetrainLoop.run` per tick | Matches batch demo; one JSONL row per tick for live Grafana |
| Post-promote monitor-only scoring | Avoids re-fitting a candidate every tick after promote |
| Email = notification side-channel | No SMTP in `RetrainLoop`; scripts call `maybe_alert_drift` after `run` |
| Significant = `should_retrain` AND breached signals >= `alert_min_triggered_signals` | Escalation is explicit; DR gate still guards promotion |

## Feature flags (added to `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `alert_email_enabled` | `bool = False` | When off, no email ever sent |
| `alert_smtp_host` | `str = ""` | SMTP server hostname |
| `alert_smtp_port` | `int = 587` | SMTP port |
| `alert_smtp_user` | `str = ""` | SMTP username |
| `alert_smtp_password` | `str = ""` | SMTP password (env only, never committed) |
| `alert_smtp_use_tls` | `bool = True` | Use STARTTLS |
| `alert_email_from` | `str = ""` | Sender address |
| `alert_email_to` | `str = ""` | Comma-separated recipients |
| `alert_min_triggered_signals` | `int = 1` | Min breached primary signals for "significant" |
| `alert_debounce_minutes` | `int = 30` | Suppress duplicate alerts within this window |

## Files to create

```
src/nba/monitoring/alerting.py       # email notifier (significant drift, debounced)
scripts/drift_demo_common.py         # shared demo path/ingest helpers
scripts/run_online_drift_demo.py     # live producer
docs/23-online-drift-demo.md         # operator runbook
decisions/2026-06-22-online-drift-demo-live-streaming.md
tests/test_online_drift_demo.py
tests/test_monitoring_alerting.py
```

## Files to change

```
src/nba/config.py                    # alert_* flags
scripts/simulate_drift_demo.py       # import from drift_demo_common
scripts/run_retrain_loop.py          # wire maybe_alert_drift
Makefile                             # online-drift-demo, metrics-exporter-demo
monitoring/grafana/dashboards/nba-ops.json  # refresh 10s
plans/README.md                      # Phase 19 row
tests/test_ethics.py                 # guard alerting.py
tests/test_config.py                 # alert defaults off
```

## `scripts/run_online_drift_demo.py`

CLI:

```bash
NBA_USE_DRIFT_MONITORING=1 uv run python scripts/run_online_drift_demo.py \
  --warmup 3000 --events-per-tick 200 --ticks 12 --tick-seconds 15 \
  --drift-mode ramp --drift-onset 4 --seed 7
```

Pure seams (unit-tested):

- `drift_spec_for_tick(tick, mode, onset, ticks) -> DriftSpec | None`
- `restamp_events(events, clock) -> list[BanditEvent]`
- `run_one_tick(...) -> TickOutcome`

## Acceptance

- [x] `NBA_USE_DRIFT_MONITORING=0` → producer is a clean no-op exit 0.
- [x] Each tick appends one `DriftReport`; Grafana panels advance per scrape.
- [x] Significant drift sends one debounced email alert; retrain promotes through DR gate.
- [x] `alert_email_enabled=False` (default) → no email; enabled-without-creds → dry-run print.
- [x] Batch demo behavior unchanged after helper extraction.
- [x] `ruff` / `pyright` clean; `pytest` green.

**Operator guide:** [docs/24-monitoring-operator-guide.md](../docs/24-monitoring-operator-guide.md)
