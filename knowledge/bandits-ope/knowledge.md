# Bandits & OPE — Knowledge

## Policies shipped

All implement `Policy` protocol (`recommend`, `action_dist`) with **full support** (every arm p > 0).

| Policy | Module | Mechanism |
|--------|--------|-----------|
| ε-greedy | `epsilon_greedy.py` | Exploit argmax q w.p. 1−ε; uniform explore w.p. ε |
| UCB | `ucb.py` | Optimism bonus on bucketed context counts + softmax |
| Thompson | `thompson.py` | Bootstrap ensemble of B LightGBM models; MC P(best arm) |

`ExploitationBaseline` = always argmax, p=1.0 — valid DM target but **no overlap** for OPE.

## OPE estimators (`ope/estimators.py`)

Given logged `(x, a, r, p)` under logging policy, estimate target policy value V(π_e):

- **IPS** — importance-weight rewards; unbiased, high variance.
- **SNIPS** — self-normalized IPS; lower variance.
- **DM** — average q̂ under π_e; low variance, model-biased.
- **DR** — DM + IPS residual correction; doubly robust. **Primary for gate.**

Guardrails: weight clipping, ESS warning when overlap is poor.

## Promotion gate (`ope/gate.py`)

Promote candidate iff:

```
DR_value − z·SE > baseline_value + min_lift
```

Reports IPS/DM disagreement as calibration smell. Conservative by design — may HOLD when point
estimate beats baseline but lower bound does not.

## Observed demo behavior (seed=7)

- Thompson often wins DR selection; ε-greedy competitive.
- Gate promotion is marginal at moderate N — HOLD is normal and correct.
- Default UCB knobs (`ucb_c=1.0`, `temp=0.25`) flatten toward uniform when q-gaps are O(0.1).

## Phase 18 monitoring metric consistency

`deployed.json` `dr_value` must be the off-policy DR estimate of the deployed policy (same
`dr()` path as `rolling_dr_drop`). Bootstrapping with on-policy mean realized reward mixed
metric types and produced spurious `rolling_dr_drop` signals until the first gate promotion.
Initial bootstrap must also hold out the DR evaluation rows from reward-model fitting (same
reference/recent gate split as promotion); scoring DR on in-sample reference rows inflated
the baseline until the first promotion refreshed it.

`deployed.json` `promoted_at` may be timezone-naive (legacy manifests). Normalize via
`store_reader.as_utc` before subtracting from aware `now` — otherwise `days_since_promote` and
`nba_deployed_model_age_days` raise `TypeError` and halt scheduled retrain evaluation.

The drift **reference** slice must also filter to labeled events strictly after `promoted_at`
(`_split_windows` in `retrain.py`). Pre-promotion rows in the reference window skew PSI,
calibration deltas, and retrain train splits while scheduled triggers already count only
post-promote events via `count_labeled_since`.

The drift **recent** slice must honor `monitor_recent_window` when enough labeled rows exist.
Do not cap `recent_n` at `len(labeled) // 2` — that shrinks drift, overlap, and gate splits
whenever total labeled count is below twice the configured window. Reserve at least one row
for reference via `min(monitor_recent_window, len(labeled) - 1)` instead.

## Phase 18 monitor cadence

`monitor_interval_events` gates `run_monitor.py` / `run_retrain_loop.py`: count labeled rows
since the latest `drift_reports.jsonl` entry. When the report stores `n_labeled_total`, cadence
uses log growth (`current_total - n_labeled_total`); if that baseline exceeds the current log
(store reset, different DB, trimmed logs), fall back to timestamp-based counting. Skip the batch
job until the count ≥ interval (default 500). `evaluate_monitor_cadence` in `monitoring/cadence.py`
is the single implementation; the Prometheus exporter surfaces `nba_monitor_*` gauges for ops
scheduling.
