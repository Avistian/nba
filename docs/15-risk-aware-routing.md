# 15 — Risk-aware routing (step by step)

> The companion build doc for [Phase 11](../plans/phase-11-risk-aware-routing.md) — ✅ **built**. It
> shows how to stop pretending every predicted prize is equally trustworthy — using uncertainty the
> repo **already produces** — with a change localized to one method. Read
> [11-improving-nba-spatio-relational-optimization.md](11-improving-nba-spatio-relational-optimization.md)
> §6. Section 6 below documents how the shipped code actually works.

This is **Upgrade 3**: low-to-medium cost, because the uncertainty source already exists.

## 1. The free uncertainty in the repo

`src/nba/bandits/thompson.py::BootstrapEnsemble` fits `B` reward models, each on a bootstrap resample
of the logs. `q_all_members(ctx)` returns a `(B, |A|)` matrix — a **distribution** of `q` per door,
not a point. Its spread *is* the model's uncertainty. Today the orchestrator's `door_profit` only ever
uses the mean (via the policy-weighted `q`), and the spread is discarded (doc 11 §6.1).

## 2. The risk-adjusted prize

Instead of pricing a door by its bare mean, subtract a penalty proportional to its spread:

\[ \text{profit}_{\text{risk}}(x_d) = \mathbb{E}[\rho(x_d)] - \kappa \cdot \text{std}[\rho(x_d)] \]

A door the model loves *on average* but disagrees about *wildly* is discounted relative to an
equally-valuable door it's *sure* about. `κ` (`risk_kappa`) tunes risk appetite, and **`κ = 0`
recovers today's behavior exactly** — so the flag is a safe no-op until you tune it.

In code, the per-member route-relevant value is the bandit-weighted `q` per ensemble member; we take
the mean and std across members:

```python
members = ensemble.q_all_members(ctx)          # (B, |A|)
w = action_dist weights                         # bandit exploration weights
v = members @ w                                  # (B,) per-member door value
prize = v.mean() - risk_kappa * v.std()
```

## 3. The more principled version: CVaR

For a stronger guarantee, optimize the **Conditional Value-at-Risk** — the average value in the worst,
say, 10% of scenarios — of *total route value* (doc 11 §6.2). This is the standard objective for
stochastic/robust orienteering and protects the rep's whole day against a few overconfident bets. It
ties directly into the scenario machinery of [Phase 13](../plans/phase-13-dynamic-stochastic-routing.md).
`mean_std` is the default; `cvar` is the opt-in extension.

## 4. Where it plugs in

This is a **localized** change. The orchestrator gains an optional `reward_ensemble` and a
`door_profit_risk` method; `plan_route` calls it only when `use_risk_aware_routing` is set. Nothing
downstream of the prize vector changes — the same `solve_tsp_profits` consumes the (now risk-adjusted)
prizes.

```mermaid
flowchart LR
    ENS["BootstrapEnsemble\n(already in repo)"] --> RA["door_profit_risk\nmean − κ·std"]
    POL["bandit action_dist"] --> RA
    RA --> OPT[solve_tsp_profits]
    OPT --> ROUTE[walkable route]
```

## 5. Proving it (doc 11 §10)

The claim is **variance reduction**: across many simulated shifts, a risk-aware route should show
**lower variance** of realized shift value at comparable mean. The test asserts exactly this with
seeded comparisons, plus the no-op check (`κ=0` equals mean pricing) and a config guard (risk routing
on without an ensemble raises clearly).

## 6. How the shipped code works (as-built)

The shipped design is section 2's formula with one deliberate refinement (recorded in
[decisions/2026-07-06-phase11-risk-aware-routing-implementation.md](../decisions/2026-07-06-phase11-risk-aware-routing-implementation.md)):

- **The mean term is `door_profit`, not the ensemble mean.** Section 2's sketch prices a door by
  `v.mean() - κ·v.std()` where `v` is the per-member value. But the deployed point estimate is the
  *full-fit* calibrated reward model (`door_profit`), and the bootstrap ensemble's mean is only
  *close* to it — so pricing off `v.mean()` made `κ=0` a **near**-no-op, not an exact one (the first
  leaderboard run drifted to +4.345 vs the +4.526 baseline). The shipped
  [`Orchestrator.door_profit_risk`](../src/nba/pipeline/orchestrator.py) therefore returns
  `self.door_profit(ctx) - risk_kappa * per_member_value.std()`: the ensemble supplies **only the
  spread**, so `κ=0` reproduces `door_profit` bit-for-bit. `tests/test_orchestrator.py` asserts the
  exact equality, and the `phase11-risk-kappa0` leaderboard row equals `baseline` exactly.
- **`per_member_value = members @ w`.** `members = reward_ensemble.q_all_members(ctx)` is `(B, |A|)`;
  `w` is the bandit's own `action_dist(ctx)` weight vector, so the spread is measured on the
  *route-relevant* value the router actually consumes, consistent with `door_profit`'s weighting.
- **`cvar` is a per-door tail mean.** With `risk_objective="cvar"`, the price is the mean of the
  worst `cvar_alpha` fraction of `per_member_value` — a pragmatic, localized CVaR. Full route-value
  CVaR over *correlated* scenarios needs the scenario machinery of
  [Phase 13](../plans/phase-13-dynamic-stochastic-routing.md) and is deferred there; `mean_std` is
  the default.
- **Dispatch + guard.** `plan_route` prices via `door_profit_risk` only when
  `use_risk_aware_routing` is set (else `door_profit`), so nothing downstream of the prize vector
  changes. The constructor raises if the flag is on but no `reward_ensemble` was supplied.
- **Threaded through the demo.** `run_demo._build_policies` already fits a `BootstrapEnsemble` for
  Thompson; it now returns it so `run_demo` passes `reward_ensemble=ensemble` to the `Orchestrator`.
  That is why the Phase 17 leaderboard (which grades via `run_demo`) can score risk pricing with no
  harness change — it already reports `realized_shift_value_std` and `_cvar`.
- **Leaderboard verdict.** Judged on downside (std / CVaR) at comparable mean per doc 11 §10; see
  the decision log and `artifacts/leaderboard.md` for the recorded `phase11-risk-kappaX` rows.
  [`notebooks/risk_aware_routing_demo.ipynb`](../notebooks/risk_aware_routing_demo.ipynb) is the
  interactive companion (κ sweep, the exact no-op, and the risk-return frontier, flat/relational).

> Next: [16-decision-focused-learning.md](16-decision-focused-learning.md) — the biggest quality win.
