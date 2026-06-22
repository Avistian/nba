"""Regression tests for ``scripts/simulate_drift_demo.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simulate_drift_demo import _load_promoted_stack  # noqa: E402

from nba.bandits.epsilon_greedy import EpsilonGreedy  # noqa: E402
from nba.config import Settings  # noqa: E402
from nba.data.simulator import generate_logs  # noqa: E402
from nba.monitoring.retrain import _write_deployed_manifest  # noqa: E402
from nba.monitoring.signals import rolling_dr_drop  # noqa: E402
from nba.ope.estimators import LoggedBatch  # noqa: E402
from nba.reward.model import RewardModel  # noqa: E402


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        deployed_model_manifest=tmp_path / "models" / "deployed.json",
    )


def test_load_promoted_stack_policy_wraps_loaded_model(tmp_path: Path) -> None:
    """After promotion, deployed_policy must wrap the newly loaded model."""
    settings = _settings(tmp_path)
    settings.ensure_dirs()
    events = generate_logs(500, settings=settings, seed=3)
    candidate = RewardModel.fit(events, settings=settings)
    candidate_dir = settings.model_dir / "candidates" / "test"
    candidate.save(candidate_dir)
    _write_deployed_manifest(
        settings=settings,
        model_dir=candidate_dir,
        dr_value=0.42,
        dr_lb=0.40,
        baseline_value=0.38,
    )

    model, policy, baseline_dr = _load_promoted_stack(
        candidate_model_dir=candidate_dir,
        settings=settings,
        baseline_dr=0.1,
    )

    assert policy._model is model
    assert isinstance(policy, EpsilonGreedy)
    assert baseline_dr == pytest.approx(0.42)


def test_rolling_dr_drop_inflated_when_policy_wraps_stale_model(tmp_path: Path) -> None:
    """Mismatching model (q_hat) and policy (pi_e) inflates rolling_dr_drop — the reported bug."""
    settings = _settings(tmp_path)
    events = generate_logs(800, settings=settings, seed=5)
    model_new = RewardModel.fit(events, settings=settings)
    model_old = RewardModel.fit(events[:300], settings=settings)
    policy_stale = EpsilonGreedy(
        model_old, epsilon=settings.epsilon, rng=np.random.default_rng(0)
    )
    policy_aligned = EpsilonGreedy(
        model_new, epsilon=settings.epsilon, rng=np.random.default_rng(0)
    )
    recent = LoggedBatch.from_events(events[-200:])
    deployed_dr = 0.15

    stale_drop = rolling_dr_drop(
        model_new, policy_stale, recent, deployed_dr=deployed_dr, settings=settings
    ).value
    aligned_drop = rolling_dr_drop(
        model_new, policy_aligned, recent, deployed_dr=deployed_dr, settings=settings
    ).value

    assert stale_drop > aligned_drop
