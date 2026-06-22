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
    _DRIFT_DEMO_DEPLOYED_MANIFEST,
    _DRIFT_DEMO_MODEL_DIR,
    _DRIFT_DEMO_MONITORING_REPORT,
    _DRIFT_DEMO_RETRAIN_AUDIT,
    _demo_db_path,
    _demo_settings,
    _load_promoted_stack,
    _persist_events,
    _score_drift,
    run_drift_demo,
)

from nba.api.store import EventStore  # noqa: E402
from nba.bandits.epsilon_greedy import EpsilonGreedy  # noqa: E402
from nba.config import Settings  # noqa: E402
from nba.data.simulator import generate_logs  # noqa: E402
from nba.monitoring.retrain import (  # noqa: E402
    RetrainLoop,
    RetrainOutcome,
    _write_deployed_manifest,  # noqa: E402
    bootstrap_deployed,
)
from nba.monitoring.signals import (  # noqa: E402
    DriftReportContext,
    build_drift_report,
    rolling_dr_drop,
)
from nba.monitoring.store_reader import read_deployed_manifest  # noqa: E402
from nba.monitoring.triggers import RetrainTrigger  # noqa: E402
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
        settings=settings,
        events=events,
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


def test_demo_settings_redirects_all_production_artifact_defaults() -> None:
    """Default Settings must route every artifact path into the drift-demo tree."""
    settings = Settings()
    demo = _demo_settings(settings)
    assert demo.db_path == _DRIFT_DEMO_DB_PATH
    assert demo.model_dir == _DRIFT_DEMO_MODEL_DIR
    assert demo.deployed_model_manifest == _DRIFT_DEMO_DEPLOYED_MANIFEST
    assert demo.monitoring_report_path == _DRIFT_DEMO_MONITORING_REPORT
    assert demo.retrain_audit_path == _DRIFT_DEMO_RETRAIN_AUDIT


def test_demo_settings_respects_explicit_artifact_overrides(tmp_path: Path) -> None:
    custom = tmp_path / "custom"
    settings = Settings(
        db_path=custom / "events.db",
        model_dir=custom / "models",
        deployed_model_manifest=custom / "models" / "deployed.json",
        monitoring_report_path=custom / "monitoring" / "drift.jsonl",
        retrain_audit_path=custom / "monitoring" / "audit.jsonl",
    )
    demo = _demo_settings(settings)
    assert demo.db_path == custom / "events.db"
    assert demo.model_dir == custom / "models"
    assert demo.deployed_model_manifest == custom / "models" / "deployed.json"
    assert demo.monitoring_report_path == custom / "monitoring" / "drift.jsonl"
    assert demo.retrain_audit_path == custom / "monitoring" / "audit.jsonl"


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


def test_retrain_loop_runs_each_frozen_shift_not_only_k_eq_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RetrainLoop.run must run after every frozen shift until promoted, not only at k==1."""
    calls: list[int] = []

    def _spy_run(self, **kwargs) -> RetrainOutcome:
        calls.append(len(kwargs.get("events", [])))
        return RetrainOutcome(
            promoted=False,
            trigger=RetrainTrigger(should_retrain=False, reasons=(), overlap_ok=True),
            candidate_metrics={},
            gate_reason="no trigger fired",
            candidate_model_dir=None,
        )

    monkeypatch.setattr(RetrainLoop, "run", _spy_run)

    settings = _settings(tmp_path)
    run_drift_demo(
        n_pre=200,
        n_post=100,
        shifts=1,
        seed=3,
        settings=settings,
        report_path=None,
    )
    assert len(calls) == 1

    calls.clear()
    run_drift_demo(
        n_pre=200,
        n_post=100,
        shifts=3,
        seed=3,
        settings=settings,
        report_path=None,
    )
    assert len(calls) == 3


def test_no_post_retrain_shifts_when_retrain_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HOLD (or no promote) must not label later shifts as post_retrain."""

    def _hold_run(self, **kwargs) -> RetrainOutcome:
        return RetrainOutcome(
            promoted=False,
            trigger=RetrainTrigger(should_retrain=True, reasons=("reward_psi",), overlap_ok=True),
            candidate_metrics={},
            gate_reason="HOLD: DR below gate",
            candidate_model_dir=None,
        )

    monkeypatch.setattr(RetrainLoop, "run", _hold_run)

    settings = _settings(tmp_path)
    report = run_drift_demo(
        n_pre=200,
        n_post=100,
        shifts=2,
        seed=3,
        settings=settings,
        report_path=None,
    )

    assert not report.promoted
    assert len(report.shift_records) == 2
    assert all(s.phase == "frozen" for s in report.shift_records)
    assert "post_retrain" not in [s.phase for s in report.shift_records]


def test_frozen_shift_signals_match_retrain_loop_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Charted drift signals must use the same windows as RetrainLoop.run."""
    captured_reports: list = []

    original_run = RetrainLoop.run

    def _capture_run(self, **kwargs):
        outcome = original_run(self, **kwargs)
        captured_reports.append(outcome.report)
        return outcome

    monkeypatch.setattr(RetrainLoop, "run", _capture_run)

    settings = _settings(tmp_path).model_copy(
        update={
            "monitor_reference_window": 200,
            "monitor_recent_window": 80,
        }
    )
    report = run_drift_demo(
        n_pre=400,
        n_post=120,
        shifts=3,
        seed=3,
        settings=settings,
        report_path=None,
    )

    frozen_records = [s for s in report.shift_records if s.phase == "frozen"]
    assert len(captured_reports) == len(frozen_records)

    for rec, drift_report in zip(frozen_records, captured_reports, strict=True):
        assert drift_report is not None
        expected = {s.name: s.value for s in drift_report.signals}
        assert rec.signals == expected
        assert rec.overlap_ok == drift_report.overlap_ok


def test_score_drift_uses_split_windows_not_fixed_shift_slice(tmp_path: Path) -> None:
    """_score_drift must honor monitor windows, not a single shift as recent."""
    settings = _settings(tmp_path).model_copy(
        update={
            "monitor_reference_window": 120,
            "monitor_recent_window": 50,
        }
    )
    settings.ensure_dirs()
    pre_events = generate_logs(300, settings=settings, seed=1)
    deployed_model, deployed_policy, _manifest, baseline_dr = bootstrap_deployed(
        settings=settings, events=pre_events
    )
    manifest = read_deployed_manifest(settings.deployed_model_manifest)
    assert manifest is not None

    post_chunk = generate_logs(80, settings=settings, seed=2)
    cumulative = list(pre_events) + list(post_chunk)

    aligned_signals, _ = _score_drift(
        model=deployed_model,
        policy=deployed_policy,
        events=cumulative,
        settings=settings,
        deployed_dr=baseline_dr,
        promoted_at=manifest.promoted_at,
    )

    legacy_ref = pre_events[-settings.monitor_reference_window :]
    legacy_recent = post_chunk
    legacy_report = build_drift_report(
        ctx=DriftReportContext(
            model=deployed_model,
            policy=deployed_policy,
            reference=LoggedBatch.from_events(legacy_ref),
            recent=LoggedBatch.from_events(legacy_recent),
            deployed_dr=baseline_dr,
        ),
        settings=settings,
    )
    legacy_signals = {s.name: s.value for s in legacy_report.signals}

    assert aligned_signals != legacy_signals


def test_run_drift_demo_uses_isolated_artifact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end demo writes only under the drift-demo tree, not production artifacts."""
    prod_db = tmp_path / "artifacts" / "events.db"
    prod_models = tmp_path / "artifacts" / "models"
    prod_deployed = prod_models / "deployed.json"
    prod_drift = tmp_path / "artifacts" / "monitoring" / "drift_reports.jsonl"
    prod_audit = tmp_path / "artifacts" / "monitoring" / "retrain_audit.jsonl"
    for path in (prod_db.parent, prod_models, prod_drift.parent):
        path.mkdir(parents=True, exist_ok=True)
    prod_db.write_text("live production log", encoding="utf-8")
    prod_deployed.write_text('{"model_dir":"production"}', encoding="utf-8")
    prod_drift.write_text('{"production": true}\n', encoding="utf-8")
    prod_audit.write_text('{"production": true}\n', encoding="utf-8")

    demo_root = tmp_path / "artifacts" / "drift_demo"
    demo_db = demo_root / "events.db"
    demo_models = demo_root / "models"
    demo_deployed = demo_models / "deployed.json"
    demo_drift = demo_root / "monitoring" / "drift_reports.jsonl"
    demo_audit = demo_root / "monitoring" / "retrain_audit.jsonl"

    import simulate_drift_demo as mod

    monkeypatch.setattr(mod, "_DEFAULT_PRODUCTION_DB_PATH", prod_db)
    monkeypatch.setattr(mod, "_DEFAULT_PRODUCTION_MODEL_DIR", prod_models)
    monkeypatch.setattr(mod, "_DEFAULT_PRODUCTION_DEPLOYED_MANIFEST", prod_deployed)
    monkeypatch.setattr(mod, "_DEFAULT_PRODUCTION_MONITORING_REPORT", prod_drift)
    monkeypatch.setattr(mod, "_DEFAULT_PRODUCTION_RETRAIN_AUDIT", prod_audit)
    monkeypatch.setattr(mod, "_DRIFT_DEMO_DB_PATH", demo_db)
    monkeypatch.setattr(mod, "_DRIFT_DEMO_MODEL_DIR", demo_models)
    monkeypatch.setattr(mod, "_DRIFT_DEMO_DEPLOYED_MANIFEST", demo_deployed)
    monkeypatch.setattr(mod, "_DRIFT_DEMO_MONITORING_REPORT", demo_drift)
    monkeypatch.setattr(mod, "_DRIFT_DEMO_RETRAIN_AUDIT", demo_audit)

    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=prod_models,
        db_path=prod_db,
        deployed_model_manifest=prod_deployed,
        monitoring_report_path=prod_drift,
        retrain_audit_path=prod_audit,
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
    assert prod_deployed.read_text(encoding="utf-8") == '{"model_dir":"production"}'
    assert prod_drift.read_text(encoding="utf-8") == '{"production": true}\n'
    assert prod_audit.read_text(encoding="utf-8") == '{"production": true}\n'

    assert demo_db.exists()
    assert demo_deployed.exists()
    assert demo_drift.exists()
    assert demo_audit.exists()
    assert demo_drift.read_text(encoding="utf-8").strip()
    assert demo_audit.read_text(encoding="utf-8").strip()

    store = EventStore(demo_db)
    try:
        assert store.decision_count() > 0
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
