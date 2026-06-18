# Phase 13 — Upgrade 5: Dynamic & stochastic routing

**Depends on:** Phase 7 (`Orchestrator.replan`), Phase 11 (risk/uncertainty). **Goal:** handle the
day as it actually unfolds — outcomes arrive, prizes are uncertain, traffic shifts. The repo already
has a solid **re-optimize-on-event** baseline (`replan`); this phase makes the prizes *stochastic*
and adds optional *anticipatory* (lookahead) replanning, both behind flags. Grounded in
[docs/11 §8](../docs/11-improving-nba-spatio-relational-optimization.md) and the build in
[docs/17](../docs/17-dynamic-stochastic-routing.md).

> **Keep it cheap** (doc 11 §8.3): most of the value of "dynamic" is already captured by
> (a) re-planning on every meaningful event (present) + (b) risk-aware prizes ([Phase 11](phase-11-risk-aware-routing.md)).
> Full MDP/anticipatory machinery is the last 10-20% and must be justified by measurement.

## Feature flags (added to `src/nba/config.py` `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `use_stochastic_prizes` | `bool = False` | Plan against the prize *distribution* (ensemble scenarios) instead of a frozen point estimate. |
| `replan_every` | `int = 0` | Re-solve after this many serviced doors; `0` => only the existing event-driven `replan` calls. |
| `use_lookahead` | `bool = False` | Anticipatory routing: sample future scenarios before committing the next leg. |
| `n_scenarios` | `int = 16` | Scenarios sampled for stochastic/lookahead planning. |
| `lookahead_horizon` | `int = 5` | Doors of lookahead when `use_lookahead`. |

## Files to modify / create

```
src/nba/pipeline/dynamic.py        # scenario sampling + lookahead/rollout helpers (new)
src/nba/pipeline/orchestrator.py   # replan() honors stochastic prizes; optional periodic replanning
src/nba/config.py                  # the flags above
tests/test_dynamic.py
```

## Stochastic prizes (`use_stochastic_prizes`)

```python
def scenario_prizes(ensemble, contexts, policy, *, n_scenarios, rng) -> np.ndarray:
    """(n_scenarios, n_doors) sampled door values: draw an ensemble member per scenario and price
    each door by its bandit-weighted q under that member. Robust to the inevitable surprises."""
```

- `replan` solves on a **scenario-robust** prize (e.g. a CVaR-style quantile across scenarios), tying
  directly into Phase 11. Falls back to the point estimate when the flag is off.

## Anticipatory replanning (`use_lookahead`)

```python
def rollout_next_leg(orchestrator, remaining, *, horizon, n_scenarios, rng) -> Route:
    """Account for the fact that we'll re-plan again later: sample future outcomes over `horizon`
    doors, evaluate candidate next legs, commit the leg with the best expected continuation
    (a tractable rollout approximation of the shift MDP, doc 11 §8.2)."""
```

- Pure approximation layer over the existing solver; OR-Tools still does the per-leg optimization.

## Periodic replanning (`replan_every`)

- The demo's shift walk already calls `replan` on events; this flag adds deterministic periodic
  re-solves so the dynamic behavior is exercised and measurable even without external events.

## Tests

`tests/test_dynamic.py`
- **Default off:** with all flags off, `replan` behaves exactly as Phase 7 (same route).
- **Stochastic robustness:** under injected prize noise across many shifts, stochastic-prize planning
  yields **lower downside** (worst-decile realized value) than point-estimate planning at comparable
  mean.
- **Lookahead never worse on average:** seeded rollout planning matches or beats greedy re-optimize on
  mean realized shift value on the test scenarios.
- **Determinism:** fixed seeds => identical scenario draws and routes.

## Leaderboard entry (lift/regression)

Records into the [Phase 17 leaderboard](phase-17-experiment-leaderboard.md): `phase13-stochastic`
(`NBA_USE_STOCHASTIC_PRIZES=1`) and `phase13-lookahead` (`NBA_USE_LOOKAHEAD=1`). Judged primarily on
**`realized_shift_value_cvar`** (downside robustness) for stochastic prizes and on the **primary
metric** for lookahead. Expected verdict **lift on CVaR / neutral-or-better on mean**; lookahead must
not regress mean value. Both clear the DR gate to count.

## Acceptance

- Dynamic features are additive on top of the existing `replan`; all flags off reproduces today.
- Stochastic prizes shrink downside risk; lookahead doesn't regress mean value — both shown in the
  demo (doc 11 §10).
- `ruff` / `pyright` clean; `pytest` green.
