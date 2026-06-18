# Phase 11 — Upgrade 3: Risk-aware routing

**Depends on:** Phase 4 (bootstrap ensemble), Phase 7 (orchestrator). **Parallelizable with:**
Phase 10. **Goal:** stop pricing every door by a bare mean. Use the **uncertainty the repo already
produces** — the bootstrap ensemble in `bandits/thompson.py` — to discount doors the model is
*unsure* about, via a risk-adjusted prize `mean - kappa*std` (optionally a CVaR objective). All
behind flags; `risk_kappa = 0.0` recovers today's behavior exactly. Grounded in
[docs/11 §6](../docs/11-improving-nba-spatio-relational-optimization.md) and the build in
[docs/15](../docs/15-risk-aware-routing.md).

> **The free uncertainty** (doc 11 §6.1): `BootstrapEnsemble.q_all_members(ctx)` returns a `(B, |A|)`
> matrix — a *distribution* of `q` per door, not a point. Today `door_profit` collapses it to a mean
> and throws the spread away. This phase spends the spread.

## Feature flags (added to `src/nba/config.py` `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `use_risk_aware_routing` | `bool = False` | Switch `plan_route` to the risk-adjusted door price. |
| `risk_kappa` | `float = 0.0` | Penalty on per-door std. `0.0` => identical to mean pricing. |
| `risk_objective` | `Literal["mean_std","cvar"] = "mean_std"` | `mean - kappa*std`, or a CVaR of route value. |
| `cvar_alpha` | `float = 0.1` | Worst-case tail fraction for the CVaR objective. |

## Files to modify / create

```
src/nba/pipeline/orchestrator.py   # door_profit_risk + flag-driven dispatch in plan_route
src/nba/config.py                  # the flags above
tests/test_orchestrator.py         # risk pricing + variance-reduction assertions
```

The orchestrator must hold an *ensemble* to compute spread. Today it takes a `QModel`; extend the
constructor to accept an **optional** `reward_ensemble: QEnsemble | None = None`. When risk routing is
on but no ensemble is supplied, raise a clear configuration error.

## `orchestrator.door_profit_risk`

```python
def door_profit_risk(self, ctx: ProspectContext) -> float:
    """Risk-adjusted door value from the bootstrap ensemble.

    members = self._reward_ensemble.q_all_members(ctx)      # (B, |A|)
    dist    = self._policy.action_dist(ctx)                  # bandit weights
    w       = np.array([dist[a] for a in ACTIONS])
    per_member_value = members @ w                           # (B,) route-relevant door value
    return mean(per_member_value) - risk_kappa * std(per_member_value)
    """
```

- `door_profit` (mean) stays the default; `plan_route` calls `door_profit_risk` only when
  `use_risk_aware_routing` is set. With `risk_kappa == 0.0` the two are numerically equal, so the flag
  is a safe no-op until tuned.
- **CVaR variant:** when `risk_objective="cvar"`, sample route-value scenarios across ensemble members
  and optimize the mean of the worst `cvar_alpha` fraction (doc 11 §6.2) — documented as the
  principled extension; `mean_std` is the default.

## Tests

`tests/test_orchestrator.py`
- **Recovers mean at kappa=0:** `door_profit_risk(ctx)` with `risk_kappa=0` equals `door_profit(ctx)`.
- **Discounts uncertainty:** a synthetic ensemble where one door has the same mean but larger spread
  than another => the high-spread door gets a lower risk price and is dropped first under a tight
  budget.
- **Variance reduction (the claim that matters, doc 11 §10):** across many simulated shifts, the
  risk-aware route shows **lower variance** of realized shift value at comparable mean than the
  mean-priced route. Asserted as a statistical (seeded) comparison in the demo-style test.
- **Config guard:** `use_risk_aware_routing=True` with no ensemble raises a clear error.

## Leaderboard entry (lift/regression)

Records into the [Phase 17 leaderboard](phase-17-experiment-leaderboard.md): `phase11-risk-kappaX`
for a swept `NBA_RISK_KAPPA`. This upgrade is about **robustness, not mean**, so it is judged on
`realized_shift_value_std` / `realized_shift_value_cvar` (downside) at **comparable mean** — the
expected verdict is a documented **neutral on the primary metric but a lift on variance/CVaR**. A
regression is a drop in mean value beyond `ope_min_lift` (κ set too high). `κ=0` must reproduce the
`baseline` row exactly (the no-op check).

## Acceptance

- Risk-aware pricing is a localized change: only `plan_route` pricing differs; nothing downstream of
  the prize vector changes.
- `risk_kappa=0.0` is a proven no-op; raising it shrinks realized-value variance in the demo.
- `ruff` / `pyright` clean; `pytest` green.
