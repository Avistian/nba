# 24 — Monitoring operator guide

> **How to use** the NBA drift monitoring stack day to day. For build context see
> [22-drift-monitoring-retrain-loop.md](22-drift-monitoring-retrain-loop.md); for the live-streaming
> demo see [23-online-drift-demo.md](23-online-drift-demo.md).

Monitoring is **off by default**. When enabled, it scores append-only logs for drift, optionally
retrains through the same DR gate as policy promotion, and exposes metrics to Grafana. The serve
API (`/recommend`, `/feedback`) is unchanged.

---

## 1. What you get

| Layer | Tool | Purpose |
|-------|------|---------|
| **Score** | `scripts/run_monitor.py` | Compare reference vs recent windows; append `DriftReport` |
| **Act** | `scripts/run_retrain_loop.py` | If triggered → fit candidate → DR gate → promote or hold |
| **Notify** | Email via `maybe_alert_drift` | Alert on **significant** drift (debounced) |
| **Visualize** | Grafana **NBA Ops** dashboard | Live signal time series + retrain timeline |
| **Metrics** | `scripts/run_metrics_exporter.py` | Prometheus text at `:9091/metrics` |

```text
EventStore (SQLite)  ──┐
drift_reports.jsonl  ──┼──> run_monitor / run_retrain_loop
retrain_audit.jsonl  ──┤         │
deployed.json        ──┘         v
                          metrics exporter (:9091)
                                    │
                                    v
                             Prometheus → Grafana
```

---

## 2. Turn monitoring on

Set env vars (or add to `.env` — **never commit SMTP passwords**):

```bash
export NBA_USE_DRIFT_MONITORING=1          # master switch
export NBA_METRICS_EXPORTER_ENABLED=1      # optional: expose /metrics
```

Everything else uses sensible defaults in `src/nba/config.py`. Key knobs:

| Env var | Default | Meaning |
|---------|---------|---------|
| `NBA_MONITOR_INTERVAL_EVENTS` | 500 | Run monitor after this many new labeled outcomes |
| `NBA_MONITOR_REFERENCE_WINDOW` | 20000 | Max events in reference slice |
| `NBA_MONITOR_RECENT_WINDOW` | 2000 | Recent window scored for drift |
| `NBA_DRIFT_REWARD_PSI_THRESHOLD` | 0.15 | Reward distribution shift trigger |
| `NBA_DRIFT_CALIBRATION_DELTA_THRESHOLD` | 0.05 | Calibration MAE increase trigger |
| `NBA_DRIFT_FEATURE_PSI_THRESHOLD` | 0.20 | Feature covariate shift trigger |
| `NBA_DRIFT_ROLLING_DR_DROP_THRESHOLD` | 0.03 | Rolling DR drop trigger |
| `NBA_RETRAIN_MIN_NEW_EVENTS` | 2000 | Min new labeled rows for scheduled retrain |
| `NBA_RETRAIN_MAX_AGE_DAYS` | 30 | Scheduled safety retrain if no drift signal |

With `NBA_USE_DRIFT_MONITORING=0`, all monitor scripts exit cleanly and the serve path is unchanged.

---

## 3. Production ops workflow

Assume the API is running and appending to `artifacts/events.db`.

### Step A — Score drift (monitor only)

```bash
cd nba
NBA_USE_DRIFT_MONITORING=1 \
  uv run python scripts/run_monitor.py --db artifacts/events.db
```

Output example:

```text
appended DriftReport to artifacts/monitoring/drift_reports.jsonl
  n_reference=1500  n_recent=500  overlap_ok=True
  reward_psi            +0.08 (thr 0.150) [ok]
  calibration_drift     +0.02 (thr 0.050) [ok]
  ...
```

- Skips if cadence not met (`monitor_interval_events`). Force with `--force`.
- Also accepts `--logs data/logs.parquet` instead of `--db`.

### Step B — Monitor + retrain + email

```bash
NBA_USE_DRIFT_MONITORING=1 \
NBA_ALERT_EMAIL_ENABLED=1 \
  uv run python scripts/run_retrain_loop.py --db artifacts/events.db
```

- **No trigger** → drift report appended, audit HOLD, no model change.
- **Trigger + DR gate pass** → candidate promoted; `deployed.json` updated atomically.
- **Trigger + DR gate fail** → audit HOLD; deployed model unchanged.
- Add `--no-email` to skip the alert. Add `--force` to bypass cadence gate.

Schedule these on cron (e.g. nightly or every N hours). The monitor decides *whether* to retrain;
you do not retrain on a fixed daily schedule.

### Step C — Expose metrics (optional)

```bash
NBA_METRICS_EXPORTER_ENABLED=1 \
  uv run python scripts/run_metrics_exporter.py
# listens on http://localhost:9091/metrics
```

One-shot dump (no HTTP server):

```bash
NBA_METRICS_EXPORTER_ENABLED=1 \
  uv run python scripts/run_metrics_exporter.py --once
```

### Step D — Grafana + Prometheus (optional, requires Docker)

```bash
make monitoring-up    # Grafana :3000, Prometheus :9090
make monitoring-down  # tear down
```

Open http://localhost:3000 → dashboard **NBA Ops** (admin/admin on first login).

---

## 4. How to recognize drift

Drift means the **recent** window (last ~2k labeled events) diverges from the **reference** window
(events since last promotion). You can see it in four places.

### A. Grafana (best visual)

Dashboard rows to watch:

| Panel | Drift sign |
|-------|------------|
| **Drift signals vs thresholds** | A signal line crosses its dashed threshold |
| **Triggered flags (1 = breach)** | Flips to `1` for `reward_psi`, `calibration_drift`, etc. |
| **Recent mean reward** | Shifts away from baseline after a world change |
| **Calibration MAE delta** | Climbs — model scores no longer match realized rewards |
| **Overlap health** | `min_p` or ESS/n below floor → **warn** (blocks promote, not retrain alone) |
| **Retrain promote count** | Increments when a candidate clears the DR gate |

### B. Terminal (script output)

`run_monitor.py` prints `[TRIGGERED]` vs `[ok]` per signal.

`run_retrain_loop.py` prints:

```text
retrain verdict: PROMOTE   # or HOLD
  trigger.should_retrain: True
  trigger.reasons: ('reward_psi',)
  gate_reason: ...
  alert: dry_run           # or sent / debounced / disabled
```

### C. Email alert (significant drift only)

Enabled with `NBA_ALERT_EMAIL_ENABLED=1`. Fires when:

- `should_retrain` is true, **and**
- at least `NBA_ALERT_MIN_TRIGGERED_SIGNALS` primary signals breached (default 1).

Without SMTP creds, the formatted email is **printed to stdout** (dry-run).

SMTP env vars:

```bash
export NBA_ALERT_SMTP_HOST=smtp.example.com
export NBA_ALERT_SMTP_PORT=587
export NBA_ALERT_SMTP_USER=alerts@example.com
export NBA_ALERT_SMTP_PASSWORD=...        # never commit
export NBA_ALERT_EMAIL_FROM=nba-alerts@example.com
export NBA_ALERT_EMAIL_TO=oncall@example.com
export NBA_ALERT_DEBOUNCE_MINUTES=30      # suppress repeat alerts
```

### D. JSONL files (headless / scripting)

```bash
# Latest drift reports (one line per monitor pass)
tail -f artifacts/monitoring/drift_reports.jsonl

# Retrain decisions
tail -f artifacts/monitoring/retrain_audit.jsonl
```

Look for `"triggered": true` inside the `signals` array.

### Signal cheat sheet

| Signal | Threshold (default) | What changed |
|--------|---------------------|--------------|
| `reward_psi` | > 0.15 | Outcome mix (season, product, territory) |
| `calibration_drift` | Δ MAE > 0.05 | `q(x,a)` no longer matches rewards |
| `feature_psi_max` | > 0.20 | Context distribution shifted |
| `rolling_dr_drop` | > 0.03 | Policy value falling on recent logs |
| `overlap_health` | min_p < 0.02 | Logs may be OPE-invalid — fix logging first |

**Retrain fires** when any primary signal breaches *or* scheduled ceiling is reached
(`retrain_max_age_days` + enough new events). Overlap failure alone does **not** trigger retrain.

---

## 5. Demo modes (learning & Grafana rehearsal)

Demos use an **isolated** tree at `artifacts/drift_demo/` — they never touch production
`artifacts/events.db` or `artifacts/models/`.

### Batch demo (fast, ~4 min)

Pre-computes all shifts; good for CI and notebooks.

```bash
make drift-demo
# writes artifacts/drift_demo_report.json
```

Then start exporter + Grafana:

```bash
make metrics-exporter-demo   # points at drift_demo tree
make monitoring-up
```

### Online demo (live Grafana animation)

Streams events over wall-clock time; one `DriftReport` per tick.

**Three terminals:**

```bash
# T1 — producer + monitor + email dry-run
make online-drift-demo

# T2 — metrics over demo tree
make metrics-exporter-demo

# T3 — Grafana
make monitoring-up
```

Faster pacing:

```bash
NBA_USE_DRIFT_MONITORING=1 NBA_ALERT_EMAIL_ENABLED=1 \
  uv run python scripts/run_online_drift_demo.py \
    --ticks 8 --tick-seconds 5 --drift-onset 2 --events-per-tick 300
```

Each tick prints:

```text
tick  5  drifting  +200 events  total=3,400  psi=+0.18  trigger=True (reward_psi)  hold  alert=dry_run
```

See [23-online-drift-demo.md](23-online-drift-demo.md) for all CLI flags.

---

## 6. Artifact map

### Production (default paths)

```
artifacts/
  events.db                              # EventStore (decisions + outcomes)
  models/
    deployed.json                        # active model manifest + DR at promote
    candidates/<timestamp>/                # promoted candidates (never in-place overwrite)
  monitoring/
    drift_reports.jsonl                  # append-only DriftReport rows
    retrain_audit.jsonl                  # append-only PROMOTE/HOLD audit
    alert_state.json                     # email debounce state (when alerting used)
```

### Demo isolation

```
artifacts/drift_demo/
  events.db
  models/deployed.json
  monitoring/drift_reports.jsonl
  monitoring/retrain_audit.jsonl
  monitoring/alert_state.json
```

When using the demo tree, point the exporter explicitly (`make metrics-exporter-demo` sets these):

```bash
NBA_DB_PATH=artifacts/drift_demo/events.db
NBA_DEPLOYED_MODEL_MANIFEST=artifacts/drift_demo/models/deployed.json
NBA_MONITORING_REPORT_PATH=artifacts/drift_demo/monitoring/drift_reports.jsonl
NBA_RETRAIN_AUDIT_PATH=artifacts/drift_demo/monitoring/retrain_audit.jsonl
```

---

## 7. Makefile quick reference

| Target | What it does |
|--------|--------------|
| `make drift-demo` | Batch drift narrative → `drift_demo_report.json` |
| `make online-drift-demo` | Live-streaming demo (15s ticks, email dry-run) |
| `make metrics-exporter` | Exporter on production artifact paths |
| `make metrics-exporter-demo` | Exporter on `artifacts/drift_demo/*` |
| `make monitoring-up` | Start Grafana + Prometheus (Docker) |
| `make monitoring-down` | Stop Grafana + Prometheus |

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Script says `use_drift_monitoring=False` | Master flag off | `export NBA_USE_DRIFT_MONITORING=1` |
| `monitor cadence not met` | Fewer than `monitor_interval_events` new labels | Wait for more feedback, or `--force` |
| Grafana panels empty | Exporter not running or wrong paths | Start exporter; use `metrics-exporter-demo` for demo tree |
| `overlap_bad` in trigger reasons | `min_p` too low for OPE | Fix logging/policy overlap before trusting retrain |
| Retrain triggers but HOLD | Candidate failed DR gate | Expected safety behavior; check `gate_reason` in audit |
| No email, `alert=disabled` | `NBA_ALERT_EMAIL_ENABLED=0` | Set to `1` |
| No email, `alert=dry_run` | Enabled but no SMTP host | Add SMTP env vars, or read stdout block |
| `alert=debounced` | Alert sent recently | Wait `alert_debounce_minutes` or clear drift episode |
| Prometheus can't scrape | Docker can't reach host :9091 | Exporter must bind `0.0.0.0:9091`; compose uses `host.docker.internal` |

### Verify metrics without Docker

```bash
NBA_METRICS_EXPORTER_ENABLED=1 \
  uv run python scripts/run_metrics_exporter.py --once | grep nba_drift
```

You should see gauges like `nba_drift_reward_psi`, `nba_drift_signal_triggered`, `nba_monitor_due`.

---

## 9. What monitoring does *not* do

- **Does not** change the hot serve path (`/recommend`, `/feedback`) — monitoring is a batch job.
- **Does not** retrain on a blind daily schedule — retrain requires evidence (signals or scheduled ceiling).
- **Does not** promote without clearing the DR gate — same bar as initial policy selection.
- **Does not** require Docker — JSONL tailing and `--once` metrics work without Grafana.

---

## 10. Related docs

| Doc | Content |
|-----|---------|
| [22-drift-monitoring-retrain-loop.md](22-drift-monitoring-retrain-loop.md) | Architecture, signals, build context |
| [23-online-drift-demo.md](23-online-drift-demo.md) | Live-streaming demo details |
| [plans/phase-18-…](../plans/phase-18-drift-monitoring-retrain-loop.md) | Full Phase 18 implementation plan |
| [plans/phase-19-…](../plans/phase-19-online-drift-demo.md) | Phase 19 online demo + email plan |
