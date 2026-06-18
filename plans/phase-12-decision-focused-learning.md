# Phase 12 — Upgrade 2: Decision-focused learning (train on route value, not prediction error)

**Depends on:** Phase 3 (reward model), Phase 6 (router), Phase 5 (OPE gate). **Goal:** close the
training/usage mismatch (doc 11 §2.1) — the reward model is graded on prediction accuracy but its
numbers are only ever used to make **include/skip/order** decisions. Train it to make the *router's*
decisions good. Ship in two on-ramps behind a flag: cheap **decision-aware row reweighting** first,
then an optional **SPO+** fine-tuning stage. The served interface (`QModel`) is unchanged, so the
orchestrator, API, bandits and OPE don't move. Grounded in
[docs/11 §5](../docs/11-improving-nba-spatio-relational-optimization.md) and the build in
[docs/16](../docs/16-decision-focused-learning.md).

> **The highest-value idea in the roadmap** (doc 11 §1, §5.5) and the genuine research seam —
> especially fused with a relational encoder ([Phase 16](phase-16-decision-focused-rdl.md)). It must
> respect the rails: no oracle at serve time, and it must clear the same DR gate as any candidate.

## Feature flags (added to `src/nba/config.py` `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `use_decision_focused` | `bool = False` | Enable decision-focused training in `RewardModel.fit`. Off => plain squared-error fit (today). |
| `df_mode` | `Literal["reweight","spo"] = "reweight"` | Cheap row-weighting, or the SPO+ fine-tune loop. |
| `df_boundary_quantile` | `float = 0.1` | Width of the include/skip boundary band that gets upweighted (reweight mode). |
| `df_upweight` | `float = 3.0` | Weight multiplier for boundary-band rows. |
| `spo_epochs` | `int = 5` | SPO+ passes over the historical neighborhoods. |
| `spo_lr` | `float = 0.01` | SPO+ subgradient step size. |
| `spo_batch` | `int = 32` | Neighborhoods per SPO+ step. |

## Files to modify / create

```
src/nba/reward/decision_focused.py   # row-weighting + SPO+ subgradient loop (new)
src/nba/reward/model.py              # optional df hook in fit(), guarded by the flag
src/nba/config.py                    # the flags above
tests/test_decision_focused.py
```

`RewardModel.fit` stays the entry point; when `use_decision_focused` is set it delegates the extra
stage to `decision_focused.py`. The returned object is still a `RewardModel` implementing `QModel`.

## On-ramp 1 — decision-aware reweighting (`df_mode="reweight"`, cheap, gradient-free)

```python
def decision_aware_weights(events, *, boundary_quantile, upweight) -> np.ndarray:
    """Per-row training weight: upweight doors near the historical include/skip boundary,
    downweight obvious includes/skips. Approximates DF learning in an afternoon (doc 11 §5.3.1).
    Boundary is estimated from the per-door bandit-weighted prize distribution on the logs."""
```

- Plugs into LightGBM's `sample_weight` — no new model, no torch, A/B-able immediately.

## On-ramp 2 — SPO+ fine-tune (`df_mode="spo"`, the real thing)

```python
def spo_finetune(model, neighborhoods, *, settings) -> RewardModel:
    """After the standard fit, for each batch of historical neighborhoods:
       1. price doors with current predicted prizes -> solve_tsp_profits -> route
       2. compute the SPO+ subgradient from realized (logged) reward vs. the route the oracle prizes
          would induce (Elmachtoub & Grigas 2017/2021)
       3. take a step on the prize predictor
    Keeps everything behind the QModel protocol; calibration re-applied after fine-tune."""
```

- **No oracle leak (doc 11 §5.4):** SPO+ uses the *true* prize only as a *training label* for regret,
  exactly like the existing `regret` metric — never inside the served model. In production the oracle
  label is replaced by the realized logged reward, which is what SPO+ is designed for.
- **Calibration retained:** the isotonic calibrator is refit after fine-tuning so DM/DR stay valid.

## Tests

`tests/test_decision_focused.py`
- **Default off:** with `use_decision_focused=False`, `fit` is byte-identical to today (same model).
- **Reweighting shifts attention:** boundary-band rows receive higher weight; obvious rows lower.
- **Regret improves (doc 11 §10):** on seeded synthetic neighborhoods, the decision-focused model
  achieves **lower decision regret** (route value vs. oracle) than the plain model at equal-or-better
  OPE value — the win SPO+ is designed to deliver.
- **Still a QModel:** the fine-tuned model satisfies the protocol and loads with the frozen feature
  schema; calibration present.
- **OPE gate unchanged:** the DF model is just another candidate; `PromotionGate` runs as-is.

## Leaderboard entry (lift/regression)

Records into the [Phase 17 leaderboard](phase-17-experiment-leaderboard.md): `phase12-reweight`
(`NBA_DF_MODE=reweight`) and `phase12-spo` (`NBA_DF_MODE=spo`), both with
`NBA_USE_DECISION_FOCUSED=1`. Judged on the **primary metric** plus **`decision_regret_mean`** (the
quantity SPO+ minimizes). Expected verdict **lift** — and it must clear the same DR gate as any value
model. This is the headline win of the roadmap, so its leaderboard row is the one to beat; a
regression means the surrogate/weights are mis-tuned.

## Acceptance

- The DF model improves realized route value / decision regret without changing any interface, and
  only promotes if it clears the existing DR lower-bound gate (doc 11 §5.4).
- No oracle symbol reaches the served model (the AST guard still passes).
- `ruff` / `pyright` clean; `pytest` green.
