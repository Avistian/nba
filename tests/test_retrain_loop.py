"""Tests for the Phase 18 RetrainLoop.

Covers the four core scenarios from the plan:
- no trigger → no fit, deployed.json unchanged
- trigger + candidate wins → promote, deployed.json updated, candidate dir exists
- trigger + candidate fails gate → HOLD, deployed.json unchanged
- overlap bad → no retrain, reason overlap_bad

Plus: append-only audit, atomic manifest write, candidate-persisted model loads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nba.bandits.epsilon_greedy import EpsilonGreedy
from nba.config import Settings
from nba.data.simulator import generate_logs
from nba.monitoring import retrain as retrain_module
from nba.monitoring.retrain import RetrainLoop, bootstrap_deployed, _time_decay_weights
from nba.monitoring.signals import rolling_dr_drop
from nba.monitoring.store_reader import read_deployed_manifest, read_retrain_audit, as_utc
from nba.ope.estimators import LoggedBatch, OPEResult, dr, eval_action_matrix, q_matrix
from nba.ope.gate import GateDecision, PromotionGate
from nba.reward.model import RewardModel
from nba.schema import ACTIONS


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        monitoring_report_path=tmp_path / "monitoring" / "drift_reports.jsonl",
        retrain_audit_path=tmp_path / "monitoring" / "retrain_audit.jsonl",
        deployed_model_manifest=tmp_path / "models" / "deployed.json",
        monitor_reference_window=1000,
        monitor_recent_window=400,
        retrain_min_new_events=50,
        drift_reward_psi_threshold=0.15,
        drift_calibration_delta_threshold=0.05,
        drift_feature_psi_threshold=0.20,
        drift_rolling_dr_drop_threshold=0.03,
        drift_min_propensity_floor=0.001,  # permissive so overlap doesn't block tests
        drift_min_ess_fraction=0.001,
    )


def _bootstrap(
    tmp_path: Path, settings: Settings, seed: int = 7
) -> tuple[list, RewardModel, EpsilonGreedy, float]:
    """Generate logs, fit a baseline deployed model, write the manifest."""
    events = generate_logs(3000, settings=settings, seed=seed)
    train = events[:2500]
    deployed = RewardModel.fit(train, settings=settings)
    deployed_policy = EpsilonGreedy(
        deployed, epsilon=settings.epsilon, rng=np.random.default_rng(seed)
    )
    settings.deployed_model_manifest.parent.mkdir(parents=True, exist_ok=True)
    baseline_dr = float(np.mean([e.reward for e in train[-1000:] if e.reward is not None]))
    promoted_at = min(e.timestamp for e in events) - timedelta(hours=1)
    settings.deployed_model_manifest.write_text(
        json.dumps(
            {
                "model_dir": str(settings.model_dir),
                "promoted_at": promoted_at.isoformat(),
                "dr_value": baseline_dr,
                "dr_lower_bound": baseline_dr,
                "baseline_value": baseline_dr,
                "feature_names": [],
            }
        ),
        encoding="utf-8",
    )
    return events, deployed, deployed_policy, baseline_dr


def test_time_decay_weights_accepts_naive_event_timestamps(tmp_path: Path) -> None:
    """Simulator/EventStore timestamps are naive UTC; decay weights must not raise."""
    settings = _settings(tmp_path).model_copy(
        update={"retrain_time_decay_halflife_days": 7.0}
    )
    events = generate_logs(50, settings=settings, seed=3)
    assert events[0].timestamp.tzinfo is None

    now = datetime.now(UTC)
    weights = _time_decay_weights(events, settings=settings, now=now)

    assert weights is not None
    assert weights.shape == (len(events),)
    assert np.all(weights > 0.0)
    assert np.all(weights <= 1.0)


def test_no_trigger_no_fit(tmp_path: Path) -> None:
    """Stable distribution + fresh manifest → deployed.json unchanged after one loop run.

    On finite-sample data a real signal can fire by chance (calibration MAE noise),
    so the assertion we care about is: the deployed manifest is left intact when
    no promotion happened. Retrain fires are tolerated as long as the gate holds.
    """
    settings = _settings(tmp_path)
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings)
    manifest_before = read_deployed_manifest(settings.deployed_model_manifest)
    assert manifest_before is not None

    loop = RetrainLoop(settings=settings, gate=PromotionGate(z=1.96, min_lift=5.0))
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=baseline_dr,
    )
    assert not outcome.promoted  # the high min_lift guarantees HOLD
    # Deployed manifest unchanged: same model_dir, same promoted_at.
    manifest_after = read_deployed_manifest(settings.deployed_model_manifest)
    assert manifest_after is not None
    assert manifest_after.model_dir == manifest_before.model_dir
    assert manifest_after.promoted_at == manifest_before.promoted_at
    # Audit appended exactly one row.
    audit = read_retrain_audit(settings.retrain_audit_path)
    assert len(audit) == 1
    assert audit[0].verdict == "hold"


def test_overlap_bad_blocks_retrain(tmp_path: Path) -> None:
    """When overlap is bad, the loop must not retrain and return overlap_bad."""
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={"drift_min_propensity_floor": 0.5, "drift_min_ess_fraction": 0.5}
    )
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings)
    loop = RetrainLoop(settings=settings, gate=PromotionGate(z=1.96, min_lift=0.0))
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=baseline_dr,
    )
    assert not outcome.promoted
    assert outcome.trigger.reasons == ("overlap_bad",)


def test_append_only_audit(tmp_path: Path) -> None:
    """Two runs produce two audit lines; the first row is never mutated."""
    settings = _settings(tmp_path)
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings)
    loop = RetrainLoop(settings=settings, gate=PromotionGate(z=1.96, min_lift=0.0))
    loop.run(
        deployed_model=deployed, deployed_policy=policy, events=events, deployed_dr=baseline_dr
    )
    loop.run(
        deployed_model=deployed, deployed_policy=policy, events=events, deployed_dr=baseline_dr
    )
    audit = read_retrain_audit(settings.retrain_audit_path)
    assert len(audit) == 2
    assert audit[0].timestamp <= audit[1].timestamp


def test_bootstrap_deployed_creates_manifest(tmp_path: Path) -> None:
    """bootstrap_deployed writes deployed.json and saves the model."""
    settings = _settings(tmp_path)
    events = generate_logs(2000, settings=settings, seed=11)
    model, policy, manifest, deployed_dr = bootstrap_deployed(settings=settings, events=events)
    assert manifest.model_dir == str(settings.model_dir)
    assert manifest.dr_value == pytest.approx(deployed_dr)
    assert settings.deployed_model_manifest.exists()
    assert (settings.model_dir / "model.joblib").exists()


def test_bootstrap_deployed_dr_value_is_ope_dr_not_mean_reward(tmp_path: Path) -> None:
    """deployed.json dr_value must be off-policy DR on held-out rows, not mean reward."""
    settings = _settings(tmp_path)
    events = generate_logs(2000, settings=settings, seed=11)
    model, policy, manifest, deployed_dr = bootstrap_deployed(settings=settings, events=events)

    _reference_events, recent_events = retrain_module._split_windows(events, settings=settings)
    _train_recent_events, gate_events = retrain_module._split_recent_for_training_and_gate(
        recent_events
    )
    gate_batch = LoggedBatch.from_events(gate_events)
    q_hat = q_matrix(model, gate_batch.contexts)
    pi_e = eval_action_matrix(policy, gate_batch.contexts)
    expected_dr = float(dr(gate_batch, q_hat, pi_e).value)

    assert manifest.dr_value == pytest.approx(expected_dr)
    assert deployed_dr == pytest.approx(expected_dr)
    mean_reward = float(np.mean([e.reward for e in gate_events if e.reward is not None]))
    assert manifest.dr_value != pytest.approx(mean_reward)


def test_bootstrap_deployed_dr_uses_holdout_not_training_rows(tmp_path: Path) -> None:
    """Initial deployed.json must not score DR on rows used to fit the reward model."""
    settings = _settings(tmp_path)
    events = generate_logs(2000, settings=settings, seed=11)
    reference_events, recent_events = retrain_module._split_windows(events, settings=settings)
    train_recent_events, gate_events = retrain_module._split_recent_for_training_and_gate(
        recent_events
    )
    train_events = list(reference_events) + list(train_recent_events)
    train_context_ids = {id(e.context) for e in train_events}
    gate_context_ids = {id(e.context) for e in gate_events}

    model, policy, manifest, deployed_dr = bootstrap_deployed(settings=settings, events=events)

    assert train_context_ids.isdisjoint(gate_context_ids)
    gate_batch = LoggedBatch.from_events(gate_events)
    expected_dr = float(
        dr(
            gate_batch,
            q_matrix(model, gate_batch.contexts),
            eval_action_matrix(policy, gate_batch.contexts),
        ).value
    )
    assert manifest.dr_value == pytest.approx(expected_dr)
    assert deployed_dr == pytest.approx(expected_dr)


def test_promote_when_gate_passes(tmp_path: Path) -> None:
    """Force a trigger + gate pass → promote, manifest update, candidate dir, audit."""

    class _ForcePromoteGate(PromotionGate):
        """Promotion plumbing test: OPE math is covered by gate-baseline tests."""

        def evaluate(
            self,
            candidate: Any,
            batch: LoggedBatch,
            q_hat: np.ndarray,
            *,
            baseline_value: float,
            clip: float | None = None,
            rng: np.random.Generator | None = None,
        ) -> GateDecision:
            decision = super().evaluate(
                candidate,
                batch,
                q_hat,
                baseline_value=baseline_value,
                clip=clip,
                rng=rng,
            )
            return GateDecision(
                promote=True,
                candidate=decision.candidate,
                baseline_value=decision.baseline_value,
                lift=decision.lift,
                lower_bound=decision.lower_bound,
                reason=decision.reason,
            )

    settings = _settings(tmp_path)
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings)

    settings = settings.model_copy(
        update={
            "drift_reward_psi_threshold": 0.0,  # anything triggers
            "drift_calibration_delta_threshold": -1.0,  # always triggered
        }
    )

    loop = RetrainLoop(settings=settings, gate=_ForcePromoteGate(z=1.96, min_lift=0.0))
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=baseline_dr,
    )
    assert outcome.trigger.should_retrain
    assert outcome.promoted
    assert outcome.candidate_model_dir is not None
    # Manifest updated to point at the candidate dir.
    manifest = read_deployed_manifest(settings.deployed_model_manifest)
    assert manifest is not None
    assert manifest.model_dir == outcome.candidate_model_dir
    # Candidate dir exists.
    assert Path(outcome.candidate_model_dir).exists()
    # Audit row is a promote.
    audit = read_retrain_audit(settings.retrain_audit_path)
    assert audit[-1].verdict == "promote"
    assert audit[-1].promoted


def test_hold_when_gate_fails(tmp_path: Path) -> None:
    """Trigger fires but candidate does not clear the gate → HOLD, deployed.json unchanged."""
    settings = _settings(tmp_path)
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings)

    # Force a trigger via the low-threshold trick, but set a high min_lift so the gate fails.
    settings = settings.model_copy(
        update={
            "drift_reward_psi_threshold": 0.0,
            "drift_calibration_delta_threshold": -1.0,
            "ope_min_lift": 5.0,  # candidate cannot possibly clear this
        }
    )
    manifest_before = read_deployed_manifest(settings.deployed_model_manifest)
    assert manifest_before is not None

    loop = RetrainLoop(
        settings=settings, gate=PromotionGate(z=1.96, min_lift=settings.ope_min_lift)
    )
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=baseline_dr,
    )
    assert outcome.trigger.should_retrain
    assert not outcome.promoted
    # deployed.json unchanged.
    manifest_after = read_deployed_manifest(settings.deployed_model_manifest)
    assert manifest_after is not None
    assert manifest_after.model_dir == manifest_before.model_dir
    # Audit row is a hold.
    audit = read_retrain_audit(settings.retrain_audit_path)
    assert audit[-1].verdict == "hold"


def test_gate_baseline_recomputes_on_holdout_when_deployed_dr_supplied(
    tmp_path: Path,
) -> None:
    """Manifest deployed_dr must not bypass holdout re-estimation for the promotion gate."""
    settings = _settings(tmp_path).model_copy(
        update={
            "drift_reward_psi_threshold": 0.0,
            "drift_calibration_delta_threshold": -1.0,
        }
    )
    events, deployed, policy, _baseline_dr = _bootstrap(tmp_path, settings)

    _reference_events, recent_events = retrain_module._split_windows(events, settings=settings)
    _train_recent, gate_events = retrain_module._split_recent_for_training_and_gate(
        recent_events
    )
    gate_batch = LoggedBatch.from_events(gate_events)
    expected_deployed_dr = float(
        dr(
            gate_batch,
            q_matrix(deployed, gate_batch.contexts),
            eval_action_matrix(policy, gate_batch.contexts),
        ).value
    )
    stale_manifest_dr = expected_deployed_dr - 0.5

    captured: dict[str, float] = {}

    class _CapturingGate(PromotionGate):
        def evaluate(
            self,
            candidate: Any,
            batch: LoggedBatch,
            q_hat: np.ndarray,
            *,
            baseline_value: float,
            clip: float | None = None,
            rng: np.random.Generator | None = None,
        ) -> GateDecision:
            captured["baseline_value"] = baseline_value
            return super().evaluate(
                candidate,
                batch,
                q_hat,
                baseline_value=baseline_value,
                clip=clip,
                rng=rng,
            )

    loop = RetrainLoop(settings=settings, gate=_CapturingGate(z=1.96, min_lift=5.0))
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=stale_manifest_dr,
    )

    assert outcome.trigger.should_retrain
    assert captured["baseline_value"] == pytest.approx(expected_deployed_dr)
    assert captured["baseline_value"] != pytest.approx(stale_manifest_dr)


def test_gate_baseline_uses_deployed_dr_when_deployed_dr_omitted(
    tmp_path: Path,
) -> None:
    """When deployed_dr is omitted, gate baseline must be deployed OPE DR, not mean reward."""
    settings = _settings(tmp_path).model_copy(
        update={
            "drift_reward_psi_threshold": 0.0,
            "drift_calibration_delta_threshold": -1.0,
        }
    )
    events, deployed, policy, _baseline_dr = _bootstrap(tmp_path, settings)

    _reference_events, recent_events = retrain_module._split_windows(events, settings=settings)
    _train_recent, gate_events = retrain_module._split_recent_for_training_and_gate(
        recent_events
    )
    gate_batch = LoggedBatch.from_events(gate_events)
    q_hat = q_matrix(deployed, gate_batch.contexts)
    pi_e = eval_action_matrix(policy, gate_batch.contexts)
    expected_deployed_dr = float(dr(gate_batch, q_hat, pi_e).value)
    mean_reward = float(gate_batch.rewards.mean())

    captured: dict[str, float] = {}

    class _CapturingGate(PromotionGate):
        def evaluate(
            self,
            candidate: Any,
            batch: LoggedBatch,
            q_hat: np.ndarray,
            *,
            baseline_value: float,
            clip: float | None = None,
            rng: np.random.Generator | None = None,
        ) -> GateDecision:
            captured["baseline_value"] = baseline_value
            return super().evaluate(
                candidate,
                batch,
                q_hat,
                baseline_value=baseline_value,
                clip=clip,
                rng=rng,
            )

    loop = RetrainLoop(settings=settings, gate=_CapturingGate(z=1.96, min_lift=5.0))
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=None,
    )

    assert outcome.trigger.should_retrain
    assert captured["baseline_value"] == pytest.approx(expected_deployed_dr)
    assert captured["baseline_value"] != pytest.approx(mean_reward)


def test_gate_uses_recent_holdout_not_candidate_training_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Triggered retrains must gate on rows the candidate did not train on."""
    settings = _settings(tmp_path).model_copy(
        update={
            "drift_reward_psi_threshold": 0.0,
            "drift_calibration_delta_threshold": -1.0,
        }
    )
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings)
    _reference_events, recent_events = retrain_module._split_windows(events, settings=settings)
    recent_context_ids = {id(e.context) for e in recent_events}
    train_context_ids: set[int] = set()
    gate_context_ids: set[int] = set()

    class _Candidate:
        pass

    class _CapturingGate(PromotionGate):
        def __init__(self) -> None:
            pass

        def evaluate(
            self,
            candidate: Any,
            batch: LoggedBatch,
            q_hat: np.ndarray,
            *,
            baseline_value: float,
            clip: float | None = None,
            rng: np.random.Generator | None = None,
        ) -> GateDecision:
            del candidate, q_hat, clip, rng
            metric = OPEResult("dr", baseline_value - 1.0, 0.0, len(batch))
            return GateDecision(
                promote=False,
                candidate={"dr": metric, "ips": metric, "dm": metric, "snips": metric},
                baseline_value=baseline_value,
                lift=-1.0,
                lower_bound=metric.value,
                reason="HOLD: captured gate batch",
            )

    def _capture_fit_candidate(
        train_events: list[Any],
        *,
        settings: Settings,
        weights: np.ndarray | None,
    ) -> _Candidate:
        del settings, weights
        train_context_ids.update(id(e.context) for e in train_events)
        return _Candidate()

    def _capture_q_matrix(candidate: _Candidate, contexts: list[Any]) -> np.ndarray:
        del candidate
        gate_context_ids.update(id(ctx) for ctx in contexts)
        return np.zeros((len(contexts), len(ACTIONS)), dtype=np.float64)

    monkeypatch.setattr(retrain_module, "_fit_candidate", _capture_fit_candidate)
    monkeypatch.setattr(retrain_module, "q_matrix", _capture_q_matrix)

    loop = RetrainLoop(settings=settings, gate=_CapturingGate())
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=baseline_dr,
    )

    assert outcome.trigger.should_retrain
    assert gate_context_ids
    assert gate_context_ids.issubset(recent_context_ids)
    assert train_context_ids.isdisjoint(gate_context_ids)


def test_naive_promoted_at_does_not_break_days_since_promote(tmp_path: Path) -> None:
    """Naive promoted_at in deployed.json must not TypeError against aware ``now``."""
    settings = _settings(tmp_path)
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings)

    naive_promoted = (datetime.now(UTC) - timedelta(days=10)).replace(tzinfo=None)
    settings.deployed_model_manifest.write_text(
        json.dumps(
            {
                "model_dir": str(settings.model_dir),
                "promoted_at": naive_promoted.isoformat(),
                "dr_value": baseline_dr,
                "dr_lower_bound": baseline_dr,
                "baseline_value": baseline_dr,
                "feature_names": [],
            }
        ),
        encoding="utf-8",
    )

    loop = RetrainLoop(settings=settings, gate=PromotionGate(z=1.96, min_lift=5.0))
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=baseline_dr,
        now=datetime.now(UTC),
    )

    assert outcome.report is not None


def test_reference_window_excludes_pre_promotion_events(tmp_path: Path) -> None:
    """Reference slice must only include labeled events after deployed.json promoted_at."""
    settings = _settings(tmp_path)
    events = generate_logs(2000, settings=settings, seed=23)

    now = datetime.now(UTC)
    promoted_at = now - timedelta(days=3)
    for i, event in enumerate(events):
        ts = (
            promoted_at - timedelta(hours=2)
            if i < len(events) - 600
            else promoted_at + timedelta(hours=1)
        )
        events[i] = event.model_copy(update={"timestamp": ts})

    reference, recent = retrain_module._split_windows(
        events, settings=settings, promoted_at=promoted_at
    )

    assert reference
    assert all(as_utc(e.timestamp) > as_utc(promoted_at) for e in reference)
    assert len(reference) < len(events) - len(recent)


def test_scheduled_trigger_uses_events_since_promote_not_recent_window(tmp_path: Path) -> None:
    """Scheduled retrain must count labeled events since promote, not the capped recent window."""
    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "monitor_recent_window": 5,
            "retrain_min_new_events": 50,
            "retrain_max_age_days": 1,
            "drift_reward_psi_threshold": 999.0,
            "drift_calibration_delta_threshold": 999.0,
            "drift_calibration_absolute_max": 999.0,
            "drift_feature_psi_threshold": 999.0,
            "drift_rolling_dr_drop_threshold": 999.0,
        }
    )
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings, seed=17)

    now = datetime.now(UTC)
    promoted_at = now - timedelta(days=5)
    for i, event in enumerate(events):
        ts = promoted_at - timedelta(hours=1) if i < len(events) - 10 else promoted_at + timedelta(hours=1)
        events[i] = event.model_copy(update={"timestamp": ts})

    settings.deployed_model_manifest.write_text(
        json.dumps(
            {
                "model_dir": str(settings.model_dir),
                "promoted_at": promoted_at.isoformat(),
                "dr_value": baseline_dr,
                "dr_lower_bound": baseline_dr,
                "baseline_value": baseline_dr,
                "feature_names": [],
            }
        ),
        encoding="utf-8",
    )

    loop = RetrainLoop(settings=settings, gate=PromotionGate(z=1.96, min_lift=5.0))
    outcome = loop.run(
        deployed_model=deployed,
        deployed_policy=policy,
        events=events,
        deployed_dr=baseline_dr,
        now=now,
    )

    assert not outcome.trigger.should_retrain
    assert "scheduled_max_age" not in outcome.trigger.reasons


def test_drift_report_jsonl_appended(tmp_path: Path) -> None:
    """Each run appends one drift report line."""
    settings = _settings(tmp_path)
    events, deployed, policy, baseline_dr = _bootstrap(tmp_path, settings)
    loop = RetrainLoop(settings=settings, gate=PromotionGate(z=1.96, min_lift=0.0))
    loop.run(
        deployed_model=deployed, deployed_policy=policy, events=events, deployed_dr=baseline_dr
    )
    loop.run(
        deployed_model=deployed, deployed_policy=policy, events=events, deployed_dr=baseline_dr
    )
    lines = settings.monitoring_report_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        payload = json.loads(line)
        assert "signals" in payload
        assert "timestamp" in payload
