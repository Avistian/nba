# 23 — Online (live-streaming) drift demo

> The companion build doc for [Phase 19](../plans/phase-19-online-drift-demo.md). It turns the
> Phase 18 batch drift narrative into a **wall-clock live stream** so you can watch drift signals
> climb on Grafana, receive an **email alert** when drift is significant, and see a gated retrain
> promote and recover — all without touching the production serve path.

> **Status: built.** See the [operator guide](24-monitoring-operator-guide.md) for day-to-day usage.

## 1. What this adds over the batch demo

| Batch (`simulate_drift_demo.py`) | Online (`run_online_drift_demo.py`) |
|----------------------------------|-------------------------------------|
| All shifts computed in memory | Events streamed tick-by-tick over wall-clock time |
| Artifacts written in one burst | One `DriftReport` per tick → live Grafana animation |
| No email alerts | Email (or dry-run print) on significant drift |
| ~4 min single run | Configurable `--tick-seconds` for pacing |

## 2. Three-terminal operator flow

### Terminal 1 — stream + monitor + alert

```bash
cd nba
make online-drift-demo
```

This runs the producer with `NBA_USE_DRIFT_MONITORING=1`, isolated demo artifacts, and a low
`monitor_interval_events` so every tick triggers the monitor.

### Terminal 2 — live metrics (demo tree)

```bash
make metrics-exporter-demo
```

Points the exporter at `artifacts/drift_demo/*` (not production paths).

### Terminal 3 — Grafana + Prometheus

```bash
make monitoring-up
# Grafana  -> http://localhost:3000  (admin/admin)
# Dashboard -> "NBA Ops" (auto-refreshes every 10s)
```

## 3. What to watch on Grafana

1. **Drift signals row** — `reward_psi`, `calibration_mae_delta`, `feature_psi_max`,
   `rolling_dr_drop` climb after `--drift-onset` ticks.
2. **Trigger annotations** — `nba_drift_signal_triggered` flips to 1 when thresholds breach.
3. **Retrain audit** — `nba_retrain_total{verdict="promote"}` increments after the DR gate passes.
4. **Recovery** — post-promote ticks show signals stabilizing.

## 4. Email alerting

Email is a **notification side-channel** — drift is still detected by Phase 18 signals/triggers.

### Enable (with real SMTP)

```bash
export NBA_ALERT_EMAIL_ENABLED=1
export NBA_ALERT_SMTP_HOST=smtp.example.com
export NBA_ALERT_SMTP_PORT=587
export NBA_ALERT_SMTP_USER=alerts@example.com
export NBA_ALERT_SMTP_PASSWORD=secret          # never commit
export NBA_ALERT_EMAIL_FROM=nba-alerts@example.com
export NBA_ALERT_EMAIL_TO=oncall@example.com
```

### Dry-run (no credentials)

When `NBA_ALERT_EMAIL_ENABLED=1` but SMTP host/user are empty, the alert is **printed to stdout**
instead of sent. Useful for the demo without a real mailbox.

### Significance + debounce

- **Significant** = `RetrainTrigger.should_retrain` AND breached primary signal count >=
  `NBA_ALERT_MIN_TRIGGERED_SIGNALS` (default 1).
- **Debounce** = suppress re-sends within `NBA_ALERT_DEBOUNCE_MINUTES` (default 30) unless drift
  clears in between.

### Manual ops path

```bash
NBA_USE_DRIFT_MONITORING=1 \
  NBA_ALERT_EMAIL_ENABLED=1 \
  uv run python scripts/run_retrain_loop.py --db artifacts/events.db
```

Add `--no-email` to skip the alert.

## 5. CLI knobs

| Flag | Default | Purpose |
|------|---------|---------|
| `--warmup` | 3000 | In-distribution events for bootstrap |
| `--events-per-tick` | 200 | Labeled events ingested per tick |
| `--ticks` | 12 | Number of live ticks |
| `--tick-seconds` | 15 | Wall-clock sleep between ticks |
| `--drift-mode` | `ramp` | `ramp` (gradual) or `step` (sudden at onset) |
| `--drift-onset` | 4 | Tick index when drift begins |
| `--seed` | 7 | Reproducibility |
| `--reset` | on | Wipe isolated demo tree before start |
| `--no-email` | off | Skip email alerts |

## 6. Isolated artifact tree

All demo output goes to `artifacts/drift_demo/`:

```
artifacts/drift_demo/
  events.db
  models/deployed.json
  monitoring/drift_reports.jsonl
  monitoring/retrain_audit.jsonl
  monitoring/alert_state.json
```

The exporter must use demo env vars (set automatically by `make metrics-exporter-demo`):

```bash
NBA_DB_PATH=artifacts/drift_demo/events.db
NBA_DEPLOYED_MODEL_MANIFEST=artifacts/drift_demo/models/deployed.json
NBA_MONITORING_REPORT_PATH=artifacts/drift_demo/monitoring/drift_reports.jsonl
NBA_RETRAIN_AUDIT_PATH=artifacts/drift_demo/monitoring/retrain_audit.jsonl
```

## 7. Fallback without Docker

```bash
# Terminal 1: producer (prints tick summaries + dry-run alerts)
make online-drift-demo

# Terminal 2: one-shot metrics dump
NBA_METRICS_EXPORTER_ENABLED=1 make metrics-exporter-demo -- --once
```

Tail `artifacts/drift_demo/monitoring/drift_reports.jsonl` for the signal time series.
