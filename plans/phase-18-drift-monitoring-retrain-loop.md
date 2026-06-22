# Phase 18 — Drift monitoring + conditional retraining loop

**Depends on:** Phase 3 (reward model), Phase 5 (OPE + promotion gate), Phase 7 (append-only event
store), Phase 8 (demo metrics), Phase 17 (leaderboard for grading the loop). **Built after Phase 17**
and **before or in parallel with Phases 10–16** — it closes the production loop the demos only
*simulate*: serve with a **frozen** model by default, **monitor** accumulated logs for drift, and
**retrain + promote only when signals fire** (or a scheduled safety ceiling is reached), always through
the same DR gate. Step-by-step build in
[docs/22](../docs/22-drift-monitoring-retrain-loop.md).

> **Not daily retrain.** A normal shift loads `artifacts/models/` and appends decisions/outcomes.
> Retraining is a **batch job triggered by evidence**, not the default morning routine. The
> `my_territory_demo` / `run_demo` "fit from scratch" path remains a teaching sandbox, not ops.

## Build sequence (where this slots in)

```
Phase 9 (relational dataset)
    -> Phase 17 (leaderboard)
        -> Phase 18 (this: monitor + conditional retrain)   <- operational ML loop
            -> Phases 10-16 (each upgrade still proves itself on the board)
```

Phase 18 reuses Phase 17's grading machinery: a retrain candidate must **clear the DR gate** vs the
currently deployed model/policy before promotion — same bar as initial policy selection.

## Problem statement

Non-stationarity (seasonality, market shifts, new neighborhoods) makes a frozen `q(x,a)` stale.
Retraining blindly every day wastes compute and risks promoting noise. The correct ops pattern:

1. **Serve** — load deployed model + policy; `recommend` / `plan_route` unchanged.
2. **Monitor** — on a schedule (every N new labeled events or nightly batch), score drift signals
   on a **reference window** (data at last promotion) vs a **recent window** (last M events).
3. **Trigger** — if any primary signal breaches its threshold *or* a scheduled ceiling fires
   (`retrain_max_age_days` with enough new data), enqueue a retrain.
4. **Retrain candidate** — `RewardModel.fit` on reference plus the older portion of the recent
   window (optional time-decay weights); refresh bootstrap ensemble if Thompson is deployed.
5. **Gate** — OPE/DR on the newest held-out recent logs excluded from candidate fitting; promote
   iff DR lower bound clears deployed baseline + `min_lift` (reuse `PromotionGate`).
6. **Audit** — append-only `retrain_audit.jsonl` records trigger reasons, metrics, gate verdict;
   never overwrite deployed artifacts in place (write candidate dir, atomic swap on promote).

## Drift signals (what to monitor)

All signals are computed **offline** on logged `(context, action, reward, propensity)` — no oracle,
no protected/geo fields. Each returns a scalar + a pass/fail vs threshold.

| Signal | Module | Definition | Default threshold | Why it matters |
|--------|--------|------------|-------------------|----------------|
| **Reward distribution shift** | `signals.reward_psi` | PSI(reward \| reference) vs PSI(reward \| recent); binned on `{-0.2,0,0.1,0.3,1.0}` ladder | `drift_reward_psi > 0.15` | Outcome mix changed (season, product, territory). |
| **Calibration degradation** | `signals.calibration_mae` | Mean `\|q(x,a_logged) − r\|` on recent labeled rows vs same metric on reference | `Δ mae > 0.05` or `recent_mae > 0.12` | Model scores no longer match realized rewards. |
| **Feature covariate shift** | `signals.feature_psi_max` | Max PSI over allow-listed context features (from `featurize(ctx, a_ref)` with a fixed reference action, or context-only slice) | `max_psi > 0.20` | The world `x` changed; q may be wrong even if rewards look stable. |
| **Overlap health** | `signals.overlap_health` | `min(propensity)` and IPS ESS fraction on recent batch vs floors | `min_p < 0.02` or `ess/n < 0.05` | Logs may be OPE-invalid; fix logging/policy before trusting retrain. |
| **Rolling policy value (optional)** | `signals.rolling_dr` | DR estimate of **current** policy on recent window vs DR at last promotion | `drop > 0.03` absolute | Direct "is the deployed stack losing value?" (needs enough labeled recent data). |

**Trigger rule (default):**

```text
retrain_triggered =
    (reward_psi > threshold)
    OR (calibration_mae_recent - calibration_mae_ref > threshold)
    OR (feature_psi_max > threshold)
    OR (rolling_dr_drop > threshold)
    OR (scheduled: days_since_promote >= retrain_max_age_days AND n_new_labeled >= retrain_min_new_events)
```

Overlap failures **warn** and block promotion until resolved; they do not alone trigger retrain
(retraining on bad logs makes things worse).

## Feature flags (added to `src/nba/config.py` `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `use_drift_monitoring` | `bool = False` | When off, no monitor/retrain path runs; serve loop unchanged. |
| `monitor_reference_window` | `int = 20_000` | Max events in the reference slice (from last promotion snapshot). |
| `monitor_recent_window` | `int = 2_000` | Recent events scored for drift. |
| `monitor_interval_events` | `int = 500` | Run monitor after this many new labeled outcomes (batch job cadence). |
| `retrain_min_new_events` | `int = 2_000` | Minimum new labeled rows before a scheduled or drift retrain is allowed. |
| `retrain_max_age_days` | `int = 30` | Scheduled safety retrain if no drift signal but data is stale. |
| `drift_reward_psi_threshold` | `float = 0.15` | Reward PSI trigger. |
| `drift_calibration_delta_threshold` | `float = 0.05` | Calibration MAE increase trigger. |
| `drift_feature_psi_threshold` | `float = 0.20` | Feature PSI trigger. |
| `drift_rolling_dr_drop_threshold` | `float = 0.03` | Rolling DR drop trigger. |
| `drift_min_propensity_floor` | `float = 0.02` | Overlap warning floor. |
| `drift_min_ess_fraction` | `float = 0.05` | ESS/n warning floor. |
| `retrain_time_decay_halflife_days` | `float \| None = None` | Optional sample weights for fit; `None` = uniform. |
| `monitoring_report_path` | `Path = artifacts/monitoring/drift_reports.jsonl` | Append-only monitor output. |
| `retrain_audit_path` | `Path = artifacts/monitoring/retrain_audit.jsonl` | Append-only retrain decisions. |
| `deployed_model_manifest` | `Path = artifacts/models/deployed.json` | Points at active model dir + promotion metrics + PSI bin edges. |
| `use_monitoring_dashboard` | `bool = False` | When on, documents/starts the optional Grafana stack; off => no Docker deps. |
| `metrics_exporter_enabled` | `bool = False` | Expose Prometheus `/metrics` from drift reports + event-store rollups. |
| `metrics_exporter_port` | `int = 9091` | HTTP port for the Prometheus text exporter. |
| `metrics_refresh_seconds` | `int = 30` | How often the exporter re-reads JSONL/SQLite between scrapes. |

With `use_drift_monitoring=False` (default), behavior is identical to today. With
`use_monitoring_dashboard=False` (default), no Docker services are required for tests or CI.

## Files to create

```
src/nba/monitoring/
  __init__.py
  signals.py           # DriftSignal, DriftReport, compute_* per signal
  triggers.py          # evaluate_triggers(report, settings) -> RetrainTrigger
  retrain.py           # RetrainLoop: candidate fit, gate, promote/skip, manifest update
  exporter.py          # Prometheus text exposition from DriftReport + audit + event rollups
  store_reader.py      # read drift_reports.jsonl / retrain_audit.jsonl / EventStore aggregates
src/nba/data/drift.py  # DriftSpec + inject drift into simulator latent (flat + relational)
monitoring/
  docker-compose.monitoring.yml     # Grafana + Prometheus (optional, dev/demo only)
  prometheus/prometheus.yml         # scrape the NBA metrics exporter
  grafana/provisioning/
    datasources/prometheus.yml
    dashboards/dashboards.yml
  grafana/dashboards/nba-ops.json   # provisioned "NBA Ops" dashboard (version-controlled)
scripts/run_monitor.py              # score drift on EventStore or parquet; append report
scripts/run_retrain_loop.py         # monitor -> trigger -> retrain -> gate -> audit
scripts/run_metrics_exporter.py     # long-lived HTTP :9091/metrics (Prometheus pull)
scripts/simulate_drift_demo.py      # multi-shift: frozen model degrades, retrain recovers
scripts/monitoring_stack.sh         # up/down helpers wrapping docker compose
notebooks/drift_retrain_demo.ipynb
tests/test_monitoring_signals.py
tests/test_retrain_loop.py
tests/test_drift_simulator.py
tests/test_monitoring_exporter.py
```

## `src/nba/monitoring/signals.py`

```python
@dataclass(frozen=True)
class DriftSignal:
    name: str
    value: float
    threshold: float
    triggered: bool
    detail: str  # human-readable, e.g. "PSI=0.21 on reward"

@dataclass(frozen=True)
class DriftReport:
    timestamp: datetime
    n_reference: int
    n_recent: int
    signals: tuple[DriftSignal, ...]
    overlap_ok: bool

def population_stability_index(ref: np.ndarray, cur: np.ndarray, *, bins: np.ndarray) -> float: ...
def reward_psi(reference: LoggedBatch, recent: LoggedBatch, *, settings) -> DriftSignal: ...
def calibration_mae(model: RewardModel, batch: LoggedBatch, *, settings) -> float: ...
def calibration_drift(model, reference, recent, *, settings) -> DriftSignal: ...
def feature_psi_max(reference, recent, *, settings) -> DriftSignal: ...
def overlap_health(recent: LoggedBatch, *, settings) -> DriftSignal: ...
def rolling_dr_drop(model, policy, recent, *, deployed_dr: float, settings) -> DriftSignal: ...

def build_drift_report(
    *,
    model: RewardModel,
    policy: Policy,
    reference: LoggedBatch,
    recent: LoggedBatch,
    deployed_dr: float | None,
    settings: Settings,
) -> DriftReport: ...
```

- PSI bins are **fixed** from training/reference quantiles (stored in `deployed.json`) so scores are
  comparable across monitor runs.
- Feature PSI uses only **allow-listed** columns from `featurize`; geo/identity never enter.

## `src/nba/monitoring/triggers.py`

```python
@dataclass(frozen=True)
class RetrainTrigger:
    should_retrain: bool
    reasons: tuple[str, ...]   # e.g. ("reward_psi", "scheduled_max_age")
    overlap_ok: bool

def evaluate_triggers(report: DriftReport, *, settings, days_since_promote: float, n_new: int) -> RetrainTrigger: ...
```

## `src/nba/monitoring/retrain.py`

```python
@dataclass(frozen=True)
class RetrainOutcome:
    promoted: bool
    trigger: RetrainTrigger
    candidate_metrics: dict[str, float]   # mse, calibration_mae, dr, dr_lb
    gate_reason: str
    candidate_model_dir: Path | None

class RetrainLoop:
    def __init__(self, *, settings, store: EventStore, gate: PromotionGate): ...

    def run(
        self,
        *,
        deployed_model: RewardModel,
        deployed_policy: Policy,
        events: list[BanditEvent],
    ) -> RetrainOutcome:
        """1) build_drift_report  2) evaluate_triggers  3) if not triggered: return
           4) fit candidate on weighted train split  5) OPE gate vs deployed on recent holdout
           6) if promote: write candidate dir + update deployed.json  7) append audit row"""
```

- **Promotion** writes to `artifacts/models/candidates/<timestamp>/` then updates
  `deployed.json` (atomic rename). Serving reloads on next process start or explicit `/health`
  manifest version bump (document in Phase 7 follow-up; not required for offline demo).
- **No in-place** `model.joblib` overwrite.

## `src/nba/data/drift.py` — simulated non-stationarity for demos

Ground-truth drift for **grading and demos only** (oracle module prefix guarded like simulator).

```python
@dataclass(frozen=True)
class DriftSpec:
    """Inject a step change partway through log generation."""
    at_fraction: float = 0.5          # fraction of events before drift kicks in
    reward_scale: float = 1.0       # multiply latent appointment/closed mass post-drift
    knock_evening_boost: float = 0.0  # add to KNOCK_NOW evening effect post-drift
    weather_slam_mult: float = 1.0    # multiply bad-weather slam probability post-drift

def apply_drift_to_latent(scores: dict[Outcome, float], *, spec: DriftSpec, event_idx: int, n: int) -> dict: ...
def generate_logs_with_drift(n, *, settings, seed, spec: DriftSpec) -> list[BanditEvent]: ...
```

Relational mode: same hook in `relational_simulator.latent_scores` when `settings.drift_spec` is set
(flag `use_simulated_drift=False` by default).

## `scripts/simulate_drift_demo.py`

End-to-end narrative (seeded, prints report):

1. Generate **pre-drift** logs (`n_pre`); train + gate; record `deployed.json` + baseline DR/regret.
2. Generate **post-drift** logs (`n_post`, `DriftSpec(at_fraction=0)` — entirely post-change world).
3. Simulate **K shifts** serving with **frozen** deployed model:
   - Track rolling calibration MAE, realized reward, regret vs oracle.
   - Run `run_monitor` after each shift → expect signals to **trigger** mid-run.
4. Run **retrain loop** once triggered → candidate must pass DR gate to promote.
5. Simulate **K more shifts** with promoted model → regret/reward recover toward pre-drift baseline.
6. Write `artifacts/drift_demo_report.json` + append Phase 17 leaderboard row.

CLI:

```bash
uv run python scripts/simulate_drift_demo.py --n-pre 15000 --n-post 8000 --shifts 6 --seed 7
```

## `notebooks/drift_retrain_demo.ipynb`

Mirrors the script with plots:

- Reward histogram reference vs recent (pre/post drift)
- Calibration MAE over shift index (frozen vs after retrain)
- Drift signal time series with threshold lines
- Regret curve: frozen → trigger → post-retrain

## `scripts/run_monitor.py` / `scripts/run_retrain_loop.py`

```bash
# Score drift on the SQLite store (or --logs parquet)
uv run python scripts/run_monitor.py --db data/events.db

# Full loop: monitor, retrain if triggered, gate, audit
uv run python scripts/run_retrain_loop.py --db data/events.db
```

Both respect `NBA_USE_DRIFT_MONITORING=1`.

## Observability dashboard (Grafana + Prometheus)

The drift loop already writes **append-only facts** (`drift_reports.jsonl`, `retrain_audit.jsonl`,
`deployed.json`, SQLite events). Phase 18 adds a **read-only observability layer** on top — the same
pattern production would use (metrics for trends, dashboards for on-call, alerts for triggers) —
without coupling Grafana into the serve hot path.

### Architecture

```text
EventStore (SQLite) ─────┐
drift_reports.jsonl ─────┼──> store_reader.py ──> exporter.py (:9091/metrics)
retrain_audit.jsonl ─────┤                              │
deployed.json ───────────┘                              v
                                                 Prometheus (scrape 15s)
                                                        │
                                                        v
                                                 Grafana "NBA Ops" dashboard
```

- **Source of truth stays JSONL/SQLite.** Grafana never mutates artifacts; it only visualizes what
  `run_monitor.py` / `run_retrain_loop.py` already wrote.
- **Prometheus pull, not push.** The exporter is a small long-lived process (or a sidecar started by
  `monitoring_stack.sh`) that re-reads the append-only files on each scrape. No new database.
- **Optional and off by default.** `use_monitoring_dashboard=False` means tests and CI never need
  Docker. The exporter can still be unit-tested by curling `/metrics` text.

### `src/nba/monitoring/exporter.py`

Expose **gauges** (latest value) and **counters** (monotonic totals) in Prometheus text format:

| Metric | Type | Source | Dashboard use |
|--------|------|--------|---------------|
| `nba_drift_reward_psi` | gauge | latest `DriftReport` | PSI vs threshold line |
| `nba_drift_calibration_mae_recent` | gauge | latest report | calibration degradation |
| `nba_drift_calibration_mae_delta` | gauge | latest report | Δ vs reference |
| `nba_drift_feature_psi_max` | gauge | latest report | covariate shift |
| `nba_drift_rolling_dr` | gauge | latest report | policy value on recent window |
| `nba_drift_rolling_dr_drop` | gauge | latest report | drop vs deployed DR |
| `nba_drift_overlap_min_propensity` | gauge | latest report | OPE overlap health |
| `nba_drift_overlap_ess_fraction` | gauge | latest report | ESS/n floor |
| `nba_drift_signal_triggered{signal="..."}` | gauge 0/1 | per signal | annotation + stat panel |
| `nba_deployed_model_age_days` | gauge | `deployed.json` | staleness |
| `nba_deployed_dr_lb` | gauge | `deployed.json` | shipped policy value |
| `nba_events_labeled_total` | counter | EventStore | log volume |
| `nba_events_recent_mean_reward` | gauge | EventStore recent window | outcome mix drift |
| `nba_retrain_total{verdict="promote\|hold"}` | counter | `retrain_audit.jsonl` | retrain timeline |

Each gauge that has a configured threshold also exports `nba_drift_<signal>_threshold` so Grafana
alert rules can compare value/threshold without hard-coding numbers in the dashboard JSON.

```python
def render_prometheus_text(*, snapshot: MonitoringSnapshot, settings: Settings) -> str: ...

def build_snapshot(
    *,
    settings: Settings,
    store: EventStore | None = None,
) -> MonitoringSnapshot:
    """Read JSONL tails + deployed manifest + optional EventStore rollups."""
```

`MonitoringSnapshot` is a plain dataclass — unit-testable without HTTP or Docker.

### `scripts/run_metrics_exporter.py`

```bash
# Start the exporter (blocks; re-reads artifacts every metrics_refresh_seconds)
uv run python scripts/run_metrics_exporter.py --port 9091

# One-shot dump for CI / debugging (no HTTP server)
uv run python scripts/run_metrics_exporter.py --once > /tmp/nba-metrics.prom
```

Honors `NBA_METRICS_EXPORTER_ENABLED=1`. When disabled, the script exits 0 with a message (no-op).

### `monitoring/docker-compose.monitoring.yml`

Optional dev stack (Grafana OSS + Prometheus). **Not** started by `make test` or the serve API.

```bash
# Terminal 1: exporter (reads local artifacts)
NBA_METRICS_EXPORTER_ENABLED=1 uv run python scripts/run_metrics_exporter.py

# Terminal 2: Grafana + Prometheus
./scripts/monitoring_stack.sh up
# Grafana -> http://localhost:3000  (admin/admin, change on first login)
# Prometheus -> http://localhost:9090
```

`scripts/monitoring_stack.sh` wraps `docker compose -f monitoring/docker-compose.monitoring.yml`
with `host.docker.internal` (or `extra_hosts`) so Prometheus inside Docker can scrape the host-bound
exporter on `:9091`.

Add Makefile targets:

```makefile
monitoring-up:
	./scripts/monitoring_stack.sh up

monitoring-down:
	./scripts/monitoring_stack.sh down
```

### Provisioned Grafana dashboard — `monitoring/grafana/dashboards/nba-ops.json`

Version-controlled dashboard **"NBA Ops"** with rows:

1. **Deployed model** — age (days), DR lower bound, last promote timestamp, overlap_ok badge.
2. **Drift signals** — time series of all five signals with threshold constant lines; red annotations
   when `nba_drift_signal_triggered==1`.
3. **Calibration & reward** — recent mean reward, calibration MAE recent vs reference, reward
   histogram (Infinity/JSON datasource reading the latest report's binned counts if exported).
4. **Overlap & OPE validity** — min propensity, ESS fraction vs floors (warn band).
5. **Retrain audit** — table/timeline of PROMOTE/HOLD rows from `retrain_audit.jsonl` (Grafana
   PostgreSQL/SQLite is overkill; export last N audit rows as a JSON array gauge or use Grafana's
   **Logs/JSON** panel fed by `exporter.py`).

During `simulate_drift_demo.py`, optionally start the exporter and print the Grafana URL so a human
can watch signals cross thresholds live while the demo runs.

### Alerting (prototype-grade)

Provision one Grafana alert rule group **"NBA drift triggers"** (paused by default in repo JSON):

- `nba_drift_reward_psi > nba_drift_reward_psi_threshold` for 2 consecutive evaluations.
- `nba_drift_calibration_mae_delta > threshold`.
- `nba_drift_overlap_min_propensity < floor` → **warning** (blocks promote, does not page for retrain).

Contact points stay empty in the repo (no Slack webhook committed). Document how to wire Slack/email
in `docs/22` §9. In production this becomes PagerDuty/Opsgenie; locally Grafana's UI notification is
enough.

### Fallback without Docker

For agents/CI that cannot run Docker:

- `run_metrics_exporter.py --once` prints Prometheus text (asserted in tests).
- `simulate_drift_demo.py` still writes `artifacts/drift_demo_report.json` and the notebook plots
  remain the zero-dependency visualization path.
- Grafana dashboard JSON is importable manually (`+ Import → upload nba-ops.json`) if Docker is
  available later.

## Tests

`tests/test_monitoring_signals.py`
- PSI is 0 on identical batches; increases when reward mix shifts artificially.
- Calibration MAE detects miscalibrated q on a shifted batch.
- Feature PSI ignores non-allow-listed fields.
- Overlap health flags `min_p` below floor.

`tests/test_retrain_loop.py`
- **No trigger → no fit** (mock events, stable distribution).
- **Trigger → fit candidate → gate**; promote only when DR lb clears baseline.
- **Append-only audit**; `deployed.json` unchanged on HOLD.
- **Overlap bad → retrain blocked** with reason.

`tests/test_drift_simulator.py`
- Pre/post drift logs have different mean reward (labeled).
- With `DriftSpec` disabled, matches standard `generate_logs` within tolerance.

`tests/test_monitoring_exporter.py`
- `build_snapshot` on synthetic `DriftReport` + audit rows produces expected gauge values.
- `render_prometheus_text` includes `# HELP` / `# TYPE` and metric names from the table above.
- Threshold companion metrics are emitted next to each signal gauge.
- `metrics_exporter_enabled=False` → `run_metrics_exporter.py --once` is a no-op exit 0.
- No oracle symbols imported by `monitoring/exporter.py`.

## Leaderboard entry (lift/regression)

Run on the **simulated drift world**:

```bash
uv run python scripts/run_experiment.py --experiment-id phase18-drift-retrain --phase 18 \\
  --set NBA_USE_DRIFT_MONITORING=1 NBA_USE_SIMULATED_DRIFT=1
```

Compare against a **frozen-model baseline** on the same drift sim (`phase18-frozen-under-drift`).
Expected: **lift** on `realized_shift_value_mean` and lower `decision_regret_mean` after the retrain
loop engages; the monitor-only row (no retrain allowed) should **regress** vs baseline under drift.

## Acceptance

- [ ] `use_drift_monitoring=False` leaves serve/demo paths byte-identical.
- [ ] `run_monitor.py` appends a `DriftReport` row with all five signals and pass/fail flags.
- [ ] Retrain runs **only** when `RetrainTrigger.should_retrain` is true (unit-tested).
- [ ] Promoted candidate clears the same `PromotionGate` as Phase 5/17; HOLD leaves `deployed.json` unchanged.
- [ ] `simulate_drift_demo.py` shows monitor firing post-drift and metric recovery after promote.
- [ ] Oracle symbols in `data/drift.py` stay out of learning modules (extend ethics guard if needed).
- [ ] `run_metrics_exporter.py --once` emits valid Prometheus text from synthetic drift artifacts (no Docker).
- [ ] `monitoring/grafana/dashboards/nba-ops.json` imports cleanly; panels plot all five drift signals with thresholds.
- [ ] `use_monitoring_dashboard=False` → `make test` and serve API require no Docker daemon.
- [ ] `ruff` / `pyright` clean; `pytest` green.
