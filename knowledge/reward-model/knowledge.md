# Reward Model — Knowledge

## What it learns

`RewardModel` (`reward/model.py`) estimates **q(x, a) = E[r | x, a]** — expected reward of action
`a` in context `x`.

- **Learner**: LightGBM regressor on `featurize(ctx, action)` vectors.
- **Calibration**: isotonic regression on held-out split (monotone map raw → observed mean reward).
- **Interface**: `q`, `q_all`, `best_action`, `save`/`load` with feature-schema guard.

## Features (`data/features.py`)

Vector = `ALLOWED_FEATURES` numeric block + weather one-hot + action one-hot.

Frozen column order in `FEATURE_NAMES`. Model persists `feature_names.json` and refuses load on
mismatch.

**Excluded by construction**: `lat`, `lon`, `address_id`, protected attributes.

## Training path

`scripts/train_reward.py` → reads parquet logs → `RewardModel.fit` → `artifacts/models/` +
`metrics.json` (MSE raw vs calibrated, reliability buckets).

## Display calibration / certainty

`notebooks/display_calibration.ipynb` motivated checking calibrated top-vs-runner-up gap before
acting (`MIN_CERTAINTY` pattern). Calibrated q supports DM/DR and routing profit units.
