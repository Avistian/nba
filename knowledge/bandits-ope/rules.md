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
recent rows. The same holdout split applies when bootstrapping the initial ``deployed.json``.

*Confirmed: retrain loop regression for in-sample DR inflation, 2026-06-22.*

## R7: Retrain gate baseline must be OPE DR on the gate holdout batch

Always re-estimate the deployed policy's DR on the same held-out rows used for the candidate.
Manifest ``deployed_dr`` is for drift monitoring (``rolling_dr_drop``) only and must not
short-circuit the promotion gate. ``PromotionGate`` compares candidate DR lower bound to
``baseline_value + min_lift``; mixing stale manifest values or mean logged reward with DR
breaks promote/hold decisions.

*Confirmed: `test_gate_baseline_uses_deployed_dr_when_deployed_dr_omitted`,
`test_gate_baseline_recomputes_on_holdout_when_deployed_dr_supplied`, 2026-06-22.*

## R8: Monitor/retrain must rebuild deployed policy from manifest metadata

When `deployed.json` exists, `run_monitor.py` and `run_retrain_loop.py` must load the policy
family recorded in the manifest (and apply `EthicalPolicy` when `ethical_wrapper` is true).
Hard-coding `EpsilonGreedy` skews `pi_e`, `rolling_dr_drop`, and promotion-gate baselines when
production serves Thompson, UCB, or an ethics wrapper.

*Confirmed: `test_load_deployed_stack_thompson_differs_from_epsilon_pi_e`, 2026-06-22.*

## R9: Thompson promotion must persist `ensemble.json`

When `policy_family` is `thompson`, the retrain promotion path must write the candidate
`BootstrapEnsemble` to the candidate `model_dir` alongside the reward model. The manifest alone
is insufficient — `load_deployed_stack` rebuilds Thompson from `ensemble.json` or refits from
events, which desynchronizes monitoring when events are unavailable.

*Confirmed: `test_promote_thompson_saves_ensemble_artifact`, 2026-06-22.*
