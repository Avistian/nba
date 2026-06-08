"""Tests for :mod:`nba.reward.model`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.stats import spearmanr

from nba.config import Settings
from nba.data.simulator import generate_logs, sample_context, true_reward
from nba.reward.model import ExploitationBaseline, RewardModel
from nba.schema import ACTIONS


@pytest.fixture(scope="module")
def trained() -> tuple[RewardModel, Settings]:
    settings = Settings(seed=7)
    events = generate_logs(6000, settings=settings, seed=7)
    model = RewardModel.fit(events, settings=settings, val_frac=0.2)
    return model, settings


def _context(seed: int):
    rng = np.random.default_rng(seed)
    return sample_context({"sale_price": 230_000.0, "year_built": 1995.0}, rng)


def test_q_all_shape_and_best_action(trained: tuple[RewardModel, Settings]) -> None:
    model, _ = trained
    ctx = _context(1)
    q = model.q_all(ctx)
    assert q.shape == (len(ACTIONS),)
    assert model.best_action(ctx) is ACTIONS[int(np.argmax(q))]


def test_save_load_roundtrip(trained: tuple[RewardModel, Settings], tmp_path) -> None:
    model, _ = trained
    model.save(tmp_path / "m")
    reloaded = RewardModel.load(tmp_path / "m")
    ctx = _context(2)
    assert np.allclose(model.q_all(ctx), reloaded.q_all(ctx))


def test_load_rejects_feature_drift(trained: tuple[RewardModel, Settings], tmp_path) -> None:
    model, _ = trained
    model.save(tmp_path / "m")
    (tmp_path / "m" / "feature_names.json").write_text('["wrong"]')
    with pytest.raises(ValueError, match="feature-name mismatch"):
        RewardModel.load(tmp_path / "m")


def test_calibration_helps_mse(trained: tuple[RewardModel, Settings]) -> None:
    from nba.data.features import featurize

    model, settings = trained
    val = generate_logs(2000, settings=settings, seed=99)
    x = np.vstack([featurize(e.context, e.action) for e in val])
    y = np.array([e.reward for e in val], dtype=np.float64)
    raw = np.asarray(model.booster.predict(x), dtype=np.float64)
    cal = model._predict(x)
    mse_raw = float(np.mean((raw - y) ** 2))
    mse_cal = float(np.mean((cal - y) ** 2))
    assert mse_cal <= mse_raw + 1e-4
    # calibrated mean prediction tracks realized mean reward
    assert cal.mean() == pytest.approx(y.mean(), abs=0.03)


def test_q_recovers_oracle_ranking(trained: tuple[RewardModel, Settings]) -> None:
    model, settings = trained
    contexts = [_context(s) for s in range(200)]
    q_best = np.array([model.q_all(c).max() for c in contexts])
    oracle_best = np.array([max(true_reward(c, a) for a in ACTIONS) for c in contexts])
    result: Any = spearmanr(q_best, oracle_best)
    rho = float(result.statistic)
    assert rho > 0.3


def test_exploitation_baseline(trained: tuple[RewardModel, Settings]) -> None:
    model, _ = trained
    baseline = ExploitationBaseline(model)
    ctx = _context(3)
    action, propensity = baseline.recommend(ctx)
    assert propensity == 1.0
    assert action is model.best_action(ctx)
    dist = baseline.action_dist(ctx)
    assert pytest.approx(sum(dist.values())) == 1.0
    assert dist[action] == 1.0
