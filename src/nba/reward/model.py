"""Calibrated supervised reward model ``q(x, a) = E[r | x, a]``.

Trained on logged ``(context, action, reward)`` tuples. LightGBM regression on the sparse reward
scale {-0.2 .. 1.0} is biased near the extremes, so an isotonic calibrator (fit on a held-out
split) monotonically maps raw predictions toward observed mean reward. This model is the DM term
in doubly-robust OPE and the score the bandit explores around.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression

from nba.config import Settings
from nba.data.features import FEATURE_NAMES, featurize, featurize_batch
from nba.schema import ACTIONS, Action, BanditEvent, ProspectContext

_MODEL_FILE = "model.joblib"
_FEATURES_FILE = "feature_names.json"


@dataclass
class RewardModel:
    """A fitted LightGBM regressor plus optional isotonic calibrator."""

    booster: LGBMRegressor
    calibrator: IsotonicRegression | None
    feature_names: list[str]

    @classmethod
    def fit(
        cls,
        events: Sequence[BanditEvent],
        *,
        settings: Settings,
        val_frac: float = 0.2,
        verbose: bool = False,
    ) -> RewardModel:
        """Fit ``q(x, a)`` on labeled events with an isotonic calibrator on a held-out split."""
        labeled = [e for e in events if e.reward is not None]
        if not labeled:
            raise ValueError("no labeled events (reward is None for all)")

        x = np.vstack([featurize(e.context, e.action) for e in labeled])
        y = np.array([e.reward for e in labeled], dtype=np.float64)

        rng = np.random.default_rng(settings.seed)
        perm = rng.permutation(len(labeled))
        n_val = max(1, int(len(labeled) * val_frac))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        booster = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=settings.seed,
            objective="regression",
            verbose=-1,
        )
        callbacks: list[Any] = [early_stopping(40, verbose=verbose)]
        if verbose:
            callbacks.append(log_evaluation(50))
        booster.fit(
            x[train_idx],
            y[train_idx],
            eval_set=[(x[val_idx], y[val_idx])],
            eval_metric="l2",
            callbacks=callbacks,
        )

        raw_val = booster.predict(x[val_idx])
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.asarray(raw_val, dtype=np.float64), y[val_idx])

        return cls(booster=booster, calibrator=calibrator, feature_names=list(FEATURE_NAMES))

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Apply the booster then the calibrator (if present)."""
        raw = np.asarray(self.booster.predict(x), dtype=np.float64)
        if self.calibrator is not None:
            return np.asarray(self.calibrator.transform(raw), dtype=np.float64)
        return raw

    def q(self, ctx: ProspectContext, action: Action) -> float:
        """Return the calibrated expected reward of ``action`` at ``ctx``."""
        return float(self._predict(featurize(ctx, action)[None, :])[0])

    def q_all(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
        """Return calibrated expected rewards for every action in one booster call."""
        return self._predict(featurize_batch(ctx, actions))

    def best_action(self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> Action:
        """Return the argmax-``q`` action at ``ctx``."""
        return actions[int(np.argmax(self.q_all(ctx, actions)))]

    def save(self, model_dir: Path) -> None:
        """Persist the booster + calibrator and the frozen feature-name list."""
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"booster": self.booster, "calibrator": self.calibrator}, model_dir / _MODEL_FILE
        )
        (model_dir / _FEATURES_FILE).write_text(json.dumps(self.feature_names))

    @classmethod
    def load(cls, model_dir: Path) -> RewardModel:
        """Load a persisted model; raises if the feature schema drifted since training."""
        feature_names = json.loads((model_dir / _FEATURES_FILE).read_text())
        if feature_names != list(FEATURE_NAMES):
            raise ValueError(
                "feature-name mismatch between persisted model and current features schema"
            )
        payload = joblib.load(model_dir / _MODEL_FILE)
        return cls(
            booster=payload["booster"],
            calibrator=payload["calibrator"],
            feature_names=feature_names,
        )


class ExploitationBaseline:
    """A zero-exploration policy that always picks ``argmax q``.

    Satisfies the bandit ``Policy`` protocol (Phase 4) so OPE can score it. Its action
    distribution is a degenerate one-hot (propensity 1.0 for the chosen arm): valid as a DM target
    but no overlap, so it is the deliberate "goes blind" cautionary baseline.
    """

    name = "exploitation_baseline"

    def __init__(self, model: RewardModel) -> None:
        self._model = model

    def recommend(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> tuple[Action, float]:
        """Return ``(argmax-q action, 1.0)``."""
        return self._model.best_action(ctx, actions), 1.0

    def action_dist(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> dict[Action, float]:
        """Return a degenerate one-hot distribution over actions."""
        best = self._model.best_action(ctx, actions)
        return {a: (1.0 if a is best else 0.0) for a in actions}
