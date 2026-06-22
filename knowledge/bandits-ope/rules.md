# Bandits & OPE — Rules

## R1: Full-support action distributions

Every serving policy must emit π(a|x) > 0 for all arms. Use `validate_dist` before returning.
`ExploitationBaseline` is eval-only / DM-only, not a logging policy.

*Confirmed: Phase 4 bandit tests, Phase 5 OPE overlap requirement.*

## R2: Primary OPE estimator is DR; gate uses lower bound

`PromotionGate` promotes on DR lower confidence bound, not point estimate. Always report
IPS/SNIPS/DM alongside for transparency.

*Confirmed: `gate.py`, `evaluate_policy.py`, demo output.*

## R3: LoggedBatch rejects non-positive propensities

OPE batch construction fails fast if any p ≤ 0. Don't patch around zeros — fix the policy.

*Confirmed: `LoggedBatch.__post_init__`, store tests.*

## R4: Tune bandit knobs in reward units

UCB `c` and softmax `temp` must be scaled to calibrated q magnitude (~0.1 gaps in this project).
Copying paper defaults without rescaling breaks UCB.

*Confirmed: ARCHITECTURE.md scale caveat, demo comparisons (Thompson/ε-greedy > default UCB).*

## R5: OPE code never imports simulator oracle

Same isolation as reward/bandits — only CARP tuples.

*Confirmed: `test_no_oracle_leak` on `ope/` package.*

## R6: Promotion gate batches must be disjoint from candidate fit rows

Do not evaluate a candidate policy with OPE/DR on rows used to fit that candidate's reward model.
For retraining, fit on reference plus an older recent train split, then gate on the newest held-out
recent rows.

*Confirmed: retrain loop regression for in-sample DR inflation, 2026-06-22.*
