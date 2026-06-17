# Reward Model — Rules

## R1: Build features only from ALLOWED_FEATURES + weather + action one-hot

Never reflect over `ProspectContext` fields. Geo/identity/protected fields cannot reach the model.

*Confirmed: `features.py` construction, `test_ethics.py` allow-list tests.*

## R2: Freeze and verify FEATURE_NAMES on save/load

`RewardModel.load` raises if persisted feature names ≠ `FEATURE_NAMES`. Prevents train/serve skew.

*Confirmed: `model.py` load guard.*

## R3: Fit isotonic calibrator on held-out split

Always calibrate q before using for routing profit or DM/DR. Report both raw and calibrated MSE in
training metrics.

*Confirmed: `RewardModel.fit`, `train_reward.py` metrics.*

## R4: Single model scores all actions via action one-hot

`q_all(ctx)` featurizes each action in `ACTIONS` order — don't train per-action models.

*Confirmed: Phase 3 design, bandit integration.*
