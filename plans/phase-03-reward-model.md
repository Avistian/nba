# Phase 3 — Reward model `q(x, a)`

**Depends on:** Phase 2. **Goal:** a calibrated supervised estimate of expected reward
`q(x, a) = E[r | x, a]` learned from logged feedback, plus a deterministic exploitation baseline
(argmax `q`) that later phases beat. This is the "DM" side of doubly-robust OPE and the score the
bandit explores around.

## Files to create

```
src/nba/reward/__init__.py
src/nba/reward/model.py
scripts/train_reward.py
tests/test_reward_model.py
```

## `src/nba/reward/model.py`

### `RewardModel`

```python
@dataclass
class RewardModel:
    booster: lgb.LGBMRegressor
    calibrator: IsotonicRegression | None
    feature_names: list[str]            # frozen from features.FEATURE_NAMES; verified on load

    @classmethod
    def fit(cls, events: Sequence[BanditEvent], *, settings, val_frac=0.2) -> "RewardModel":
        # X = featurize(ctx, action) per labeled event (reward is not None); y = reward
        # LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
        #               min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
        #               random_state=settings.seed, objective="regression")
        # fit on train split with early_stopping on val (val_frac, seeded split)
        # calibrator: IsotonicRegression(out_of_bounds="clip") fit on (val preds → val rewards)
    def q(self, ctx: ProspectContext, a: Action) -> float: ...
    def q_all(self, ctx: ProspectContext, actions=ACTIONS) -> np.ndarray:   # uses featurize_batch
    def best_action(self, ctx) -> Action: ...                               # argmax q_all
    def save(self, model_dir: Path) -> None:                                # joblib + feature_names.json
    @classmethod
    def load(cls, model_dir: Path) -> "RewardModel":                        # asserts feature_names match
```

- **Calibration:** raw LightGBM regression on sparse {−0.2…1.0} rewards is biased near the
  extremes; isotonic on a held-out split monotonically maps predictions toward observed mean
  reward. `q()` applies `calibrator.transform` when present.
- **Feature-name guard:** `load` raises if persisted `feature_names != features.FEATURE_NAMES`
  (catches schema drift between training and serving).
- **Vectorized path:** `q_all` calls `featurize_batch` then a single `booster.predict` — used by
  bandits and router on every decision, so it must be O(1) booster call, not a Python loop.

### `ExploitationBaseline` (the "goes blind" comparison policy)

```python
class ExploitationBaseline:
    """Always argmax q — zero exploration. Satisfies the Policy protocol (Phase 4) so OPE can
       score it. propensity is 1.0 for the chosen arm (deterministic)."""
    def __init__(self, model: RewardModel): ...
    def recommend(self, ctx, actions=ACTIONS) -> tuple[Action, float]:   # (best, 1.0)
    def action_dist(self, ctx, actions=ACTIONS) -> dict[Action, float]:  # degenerate one-hot
```
- Note: a degenerate (no-overlap) policy is fine as an OPE *target* under DM but breaks IPS;
  this is intentionally the cautionary baseline.

## `scripts/train_reward.py`

- CLI: `--logs data/logs.parquet --out artifacts/models --val-frac 0.2 --seed 7`.
- Loads parquet → reconstructs `BanditEvent`s (or featurizes directly), fits, evaluates, saves.
- Prints metrics on the val split: MSE, MAE, and a **reliability curve** summary (bucketed
  predicted vs realized reward) before/after calibration; saves `artifacts/models/metrics.json`.

## Tests

`tests/test_reward_model.py`
- **Shape/contract:** `q_all(ctx)` length == `len(ACTIONS)`; `best_action` == `argmax q_all`.
- **Round-trip:** `save` then `load` → identical `q_all` predictions; feature-name mismatch on
  load raises.
- **Calibration helps:** on a seeded simulator val split, calibrated MSE ≤ raw MSE (allow tiny
  epsilon); calibrated mean prediction ≈ realized mean reward within tolerance.
- **Learns signal:** on simulator logs, `q` correlates positively (Spearman > 0.3) with
  `true_reward` on a fresh sample — i.e., it recovers oracle ranking, never reads the oracle.
- **Speed:** fits on 5k events in seconds (smoke timing, not a hard assert).
- **Baseline:** `ExploitationBaseline.recommend` returns propensity 1.0 and the argmax arm.

## Acceptance

- `train_reward.py` produces `artifacts/models/{model.joblib, feature_names.json, metrics.json}`.
- Calibration improves (or matches) held-out MSE and reliability; `save`==`load` predictions.
- `q` recovers the oracle's action ranking without importing any oracle symbol.
- `ruff`/`pyright` clean; `pytest` green.
