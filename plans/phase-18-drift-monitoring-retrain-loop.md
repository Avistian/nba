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
4. **Retrain candidate** — `RewardModel.fit` on reference∪recent (optional time-decay weights);
   refresh bootstrap ensemble if Thompson is deployed.
5. **Gate** — OPE/DR on held-out recent logs; promote iff DR lower bound clears deployed baseline +
   `min_lift` (reuse `PromotionGate`).
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
| `deployed_model_manifest` | `Path = artifacts/models/deployed.json` | Points at active model dir + promotion timestamp + reference metrics. |

With `use_drift_monitoring=False` (default), behavior is identical to today.

## Files to create

```
src/nba/monitoring/
  __init__.py
  signals.py           # DriftSignal, DriftReport, compute_* per signal
  triggers.py          # evaluate_triggers(report, settings) -> RetrainTrigger
  retrain.py           # RetrainLoop: candidate fit, gate, promote/skip, manifest update
src/nba/data/drift.py  # DriftSpec + inject drift into simulator latent (flat + relational)
scripts/run_monitor.py           # score drift on EventStore or parquet; append report
scripts/run_retrain_loop.py      # monitor -> trigger -> retrain -> gate -> audit
scripts/simulate_drift_demo.py   # multi-shift: frozen model degrades, retrain recovers
notebooks/drift_retrain_demo.ipynb
tests/test_monitoring_signals.py
tests/test_retrain_loop.py
tests/test_drift_simulator.py
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
- [ ] `ruff` / `pyright` clean; `pytest` green.
