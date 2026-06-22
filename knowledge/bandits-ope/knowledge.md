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

`deployed.json` `promoted_at` may be timezone-naive (legacy manifests). Normalize via
`store_reader.as_utc` before subtracting from aware `now` — otherwise `days_since_promote` and
`nba_deployed_model_age_days` raise `TypeError` and halt scheduled retrain evaluation.
