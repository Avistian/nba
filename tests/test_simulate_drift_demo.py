"""Regression tests for ``scripts/simulate_drift_demo.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simulate_drift_demo import (  # noqa: E402
    _DEFAULT_PRODUCTION_DB_PATH,
    _DRIFT_DEMO_DB_PATH,
    _demo_db_path,
    _load_promoted_stack,
    _persist_events,
    run_drift_demo,
)

from nba.api.store import EventStore  # noqa: E402
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
    policy_stale = EpsilonGreedy(model_old, epsilon=settings.epsilon, rng=np.random.default_rng(0))
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


def test_demo_db_path_redirects_away_from_production_default() -> None:
    """Drift demo must not use the shared API EventStore unless explicitly overridden."""
    settings = Settings()
    assert settings.db_path == _DEFAULT_PRODUCTION_DB_PATH
    assert _demo_db_path(settings) == _DRIFT_DEMO_DB_PATH


def test_demo_db_path_respects_explicit_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom.db"
    settings = Settings(db_path=custom)
    assert _demo_db_path(settings) == custom


def test_persist_events_does_not_unlink_production_db(tmp_path: Path) -> None:
    """_persist_events must never delete the shared production EventStore."""
    prod_db = tmp_path / "artifacts" / "events.db"
    prod_db.parent.mkdir(parents=True)
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=prod_db,
    )
    existing = generate_logs(20, settings=settings, seed=1)
    store = EventStore(prod_db)
    try:
        store.ingest_bandit_events(existing, policy_name="production")
        count_before = store.decision_count()
    finally:
        store.close()

    new_events = generate_logs(30, settings=settings, seed=2)
    _persist_events(settings, new_events)

    store = EventStore(prod_db)
    try:
        assert store.decision_count() > count_before
    finally:
        store.close()


def test_persist_events_resets_only_demo_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import simulate_drift_demo as mod

    demo_db = tmp_path / "drift_demo" / "events.db"
    demo_db.parent.mkdir(parents=True)
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=demo_db,
    )
    stale = generate_logs(5, settings=settings, seed=0)
    store = EventStore(demo_db)
    try:
        store.ingest_bandit_events(stale, policy_name="old")
    finally:
        store.close()

    monkeypatch.setattr(mod, "_DRIFT_DEMO_DB_PATH", demo_db)
    fresh = generate_logs(40, settings=settings, seed=2)
    _persist_events(settings, fresh)

    expected = len([e for e in fresh if e.reward is not None and e.outcome is not None])
    store = EventStore(demo_db)
    try:
        assert store.decision_count() == expected
    finally:
        store.close()


def test_run_drift_demo_uses_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end demo writes to the drift-demo store, leaving production db untouched."""
    prod_db = tmp_path / "artifacts" / "events.db"
    prod_db.parent.mkdir(parents=True)
    prod_db.write_text("live production log", encoding="utf-8")
    demo_db = tmp_path / "artifacts" / "drift_demo" / "events.db"

    import simulate_drift_demo as mod

    monkeypatch.setattr(mod, "_DEFAULT_PRODUCTION_DB_PATH", prod_db)
    monkeypatch.setattr(mod, "_DRIFT_DEMO_DB_PATH", demo_db)

    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=prod_db,
        deployed_model_manifest=tmp_path / "models" / "deployed.json",
    )
    report = run_drift_demo(
        n_pre=200,
        n_post=100,
        shifts=2,
        seed=3,
        settings=settings,
        report_path=None,
    )

    assert report.shift_records
    assert prod_db.read_text(encoding="utf-8") == "live production log"
    assert demo_db.exists()
    store = EventStore(demo_db)
    try:
        assert store.decision_count() > 0
    finally:
        store.close()
