## Decision: Phase 11 risk-aware routing prices `door_profit − κ·std_ensemble` — the mean term is the deployed point estimate (not the ensemble mean) so `κ=0` is a bit-exact no-op, with a pragmatic per-door CVaR variant and the whole change localized to one orchestrator method

## Context

Phase 11 (Upgrade 3) stops pricing every door by a bare mean and spends the uncertainty the repo
already produces — the bootstrap ensemble behind Thompson sampling. A door the model loves on average
but disagrees about wildly should be discounted relative to an equally-valuable door it is sure about.
It must stay off by default and reproduce today's routes exactly until a flag is set. Several
implementation forks had to be resolved, and one broke the no-op guarantee on the first run.

## Alternatives considered

- **Mean anchor for `mean_std`:** the ensemble mean (`per_member_value.mean()`, as doc 15's sketch
  literally shows) vs the deployed point estimate `door_profit` (the full-fit calibrated reward
  model). The ensemble mean only *approximates* the full-fit model.
- **CVaR scope:** a full route-value CVaR over correlated ensemble scenarios vs a pragmatic per-door
  tail mean.
- **Ensemble plumbing:** refit an ensemble inside the orchestrator vs inject the one `run_demo`
  already fits for Thompson.
- **Leaderboard params:** a fresh sweep scale vs matching the existing `baseline` row exactly.

## Reasoning

- **Anchor the mean on `door_profit`, not the ensemble mean.** Pricing `mean_std` off
  `per_member_value.mean()` made `κ=0` a *near* no-op: the first `phase11-risk-kappa0` leaderboard run
  drifted to realized value **+4.345 vs the +4.526 baseline** (a spurious regression) because the
  bootstrap ensemble's mean is only close to the full-fit model. The shipped
  `Orchestrator.door_profit_risk` returns `self.door_profit(ctx) - risk_kappa *
  per_member_value.std()` — the ensemble supplies **only the spread** — so `κ=0` reproduces
  `door_profit` bit-for-bit. Re-running gave `phase11-risk-kappa0` = **+4.526, byte-identical to
  baseline** (verdict neutral), and `tests/test_orchestrator.py` asserts the exact equality.
- **Per-door CVaR now; route-value CVaR deferred to Phase 13.** `risk_objective="cvar"` prices a door
  by the mean of its worst `cvar_alpha` fraction of per-member values — coherent, localized, and free
  of the scenario-correlation machinery. True route-value CVaR belongs with Phase 13's
  dynamic/stochastic scenarios; `mean_std` stays the default.
- **Inject the existing ensemble.** The orchestrator takes an optional `reward_ensemble: QEnsemble |
  None`; `run_demo._build_policies` now returns the `BootstrapEnsemble` it already fits for Thompson,
  and `run_demo` passes it through. So the Phase 17 leaderboard grades risk pricing with **no harness
  change** (it already reports `realized_shift_value_std` / `_cvar`). A config guard raises if the
  flag is on without an ensemble.
- **Match the baseline params** (6 shifts × seed 7, n_logs 3000, shift 40) so the rows are comparable
  to the existing `baseline`/`phase10-*` rows.

## Leaderboard results (6 shifts × seed 7, n_logs 3000, shift 40, vs `baseline` +4.526)

| experiment | realized value | Δ value | std | CVaR | verdict |
|---|---|---|---|---|---|
| baseline | +4.526 | +0.000 | 0.349 | 4.243 | neutral (reference) |
| phase11-risk-kappa0 (`NBA_RISK_KAPPA=0`) | +4.526 | +0.000 | 0.349 | 4.243 | **neutral** (exact no-op) |
| phase11-risk-kappa05 (`NBA_RISK_KAPPA=0.5`) | +3.913 | -0.614 | 0.506 | 3.234 | **regression** |
| phase11-risk-kappa10 (`NBA_RISK_KAPPA=1.0`) | +2.350 | -2.177 | 1.062 | 1.054 | **regression** |

- **`κ=0` => exact no-op:** realized value, std, and CVaR all byte-identical to baseline — the
  feature-flag safety guarantee, proven end-to-end (not just in a unit test).
- **`κ>0` => regression at single-block scale:** like Phase 10's `team2` row, the graded demo routes a
  single dense 0.3 km block whose realized-value variance is driven by *which doors are sampled and
  their outcomes*, **not** by the model's epistemic uncertainty. Discounting epistemic-uncertain doors
  there just sheds profitable doors (mean falls) and, by serving fewer doors, even *raises* realized
  variance — the opposite of the intended win. Risk-aware routing's value regime is a
  **binding-capacity, multi-cluster territory** where the served set must be chosen and caution buys
  downside protection; that regime is exercised in
  `notebooks/risk_aware_routing_demo.ipynb`, and the *mechanics* (exact no-op, uncertainty discount,
  high-spread-door-dropped-first, per-door CVaR) are proven by `tests/test_orchestrator.py`.
- Adoption stays **off by default** (the regression correctly blocks default-on).

## Trade-offs accepted

- The single-block demo shows the value story as a regression, so — as with Phase 10 — the unit tests
  (not the board) are the proof of correct mechanics, and the flag remains opt-in.
- `mean_std` uses a symmetric std penalty (penalizes upside surprises too); the `cvar` objective is
  the opt-in downside-only alternative.
- Per-door CVaR ignores cross-door correlation; the correlated route-value CVaR is deferred to
  Phase 13.

## Supersedes

None. First Phase 11 decision; builds on the Phase 4 bootstrap ensemble and the Phase 6/10 router.
