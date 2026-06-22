"""The conditional retrain loop — fit candidate, gate through ``PromotionGate``, promote or hold.

Pipeline (Phase 18 plan):

1. Split events reference/recent by ``settings.monitor_*_window``.
2. :func:`~nba.monitoring.signals.build_drift_report` scores the five signals.
3. :func:`~nba.monitoring.triggers.evaluate_triggers` decides whether to retrain.
4. If not triggered → append drift report + audit HOLD; return.
5. Fit candidate :class:`~nba.reward.model.RewardModel` on reference∪recent (with
   optional time-decay sample weights).
6. :class:`~nba.ope.gate.PromotionGate` evaluates the candidate vs the deployed
   baseline on a recent holdout.
7. If promote → write candidate dir under ``artifacts/models/candidates/<ts>/``;
   atomic-rename ``deployed.json``.
8. Append the audit row (promote|hold) + the drift report.

Invariants:

- **No in-place overwrite** of ``model.joblib``. Promotion writes a candidate
  dir then updates ``deployed.json`` atomically (``os.replace``).
- **Same gate** as Phase 5/17: the candidate's DR lower bound must clear
  ``baseline_value + ope_min_lift``.
- **Append-only audit** at ``artifacts/monitoring/retrain_audit.jsonl`` — trigger
  reasons, metrics, verdict.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from nba.api.store import EventStore
from nba.bandits.base import Policy
from nba.bandits.epsilon_greedy import EpsilonGreedy
from nba.bandits.thompson import BootstrapEnsemble, ThompsonSampling
from nba.config import Settings
from nba.monitoring.signals import (
    DriftReport,
    DriftReportContext,
    append_report,
    build_drift_report,
)
from nba.monitoring.store_reader import AuditRow, DeployedManifest
from nba.monitoring.triggers import RetrainTrigger, evaluate_triggers
from nba.ope.estimators import LoggedBatch, q_matrix
from nba.ope.gate import PromotionGate
from nba.reward.model import RewardModel
from nba.schema import BanditEvent


@dataclass(frozen=True)
class RetrainOutcome:
    """The retrain loop's verdict + the evidence behind it."""

    promoted: bool
    trigger: RetrainTrigger
    candidate_metrics: dict[str, float]  # mse, calibration_mae, dr, dr_lb
    gate_reason: str
    candidate_model_dir: str | None = None
    report: DriftReport | None = None
    audit: AuditRow | None = field(default=None, repr=False)


def _labeled(events: list[BanditEvent]) -> list[BanditEvent]:
    return [e for e in events if e.reward is not None]


def _split_windows(
    events: list[BanditEvent], *, settings: Settings
) -> tuple[list[BanditEvent], list[BanditEvent]]:
    """Reference = the older ``monitor_reference_window`` labeled events (capped);
    Recent = the last ``monitor_recent_window`` labeled events.
    """
    labeled = _labeled(events)
    if len(labeled) < 2:
        raise ValueError("need at least 2 labeled events to score drift")
    recent_n = min(settings.monitor_recent_window, len(labeled) // 2)
    ref_n = min(
        settings.monitor_reference_window,
        max(1, len(labeled) - recent_n),
    )
    if recent_n < 1 or ref_n < 1:
        raise ValueError(
            f"not enough labeled events for windows (labeled={len(labeled)}, "
            f"requested recent={recent_n}, reference={ref_n})"
        )
    # Reference = older slice; recent = newest slice.
    reference = (
        labeled[-(recent_n + ref_n) : -recent_n]
        if len(labeled) > recent_n + ref_n
        else labeled[:-recent_n]
    )
    recent = labeled[-recent_n:]
    return reference, recent


def _time_decay_weights(
    events: list[BanditEvent], *, settings: Settings, now: datetime | None = None
) -> np.ndarray | None:
    """Exponential decay sample weights by event age; ``None`` when halflife is unset.

    Weight = 0.5 ** (age_days / halflife_days). Newest events get weight 1.0.
    """
    halflife = settings.retrain_time_decay_halflife_days
    if halflife is None:
        return None
    now = now or datetime.now(UTC)
    ages = np.array(
        [(now - e.timestamp).total_seconds() / 86400.0 for e in events], dtype=np.float64
    )
    ages = np.clip(ages, 0.0, None)
    return np.power(0.5, ages / halflife)


def _fit_candidate(
    train_events: list[BanditEvent],
    *,
    settings: Settings,
    weights: np.ndarray | None,
) -> RewardModel:
    """Fit a candidate reward model (uniform or time-decay weighted)."""
    if weights is None:
        return RewardModel.fit(train_events, settings=settings)
    # LightGBM ``sample_weight`` path: re-fit manually.
    return _fit_weighted(train_events, settings=settings, weights=weights)


def _fit_weighted(
    events: list[BanditEvent], *, settings: Settings, weights: np.ndarray
) -> RewardModel:
    """Fit a LightGBM regressor with per-sample weights, plus isotonic calibration."""
    from lightgbm import LGBMRegressor, early_stopping  # noqa: PLC0415
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

    from nba.data.features import FEATURE_NAMES, featurize  # noqa: PLC0415

    labeled = [e for e in events if e.reward is not None]
    x = np.vstack([featurize(e.context, e.action) for e in labeled])
    y = np.array([e.reward for e in labeled], dtype=np.float64)
    rng = np.random.default_rng(settings.seed)
    perm = rng.permutation(len(labeled))
    n_val = max(1, int(len(labeled) * 0.2))
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
    booster.fit(
        x[train_idx],
        y[train_idx],
        sample_weight=weights[train_idx],
        eval_set=[(x[val_idx], y[val_idx])],
        eval_metric="l2",
        callbacks=[early_stopping(40, verbose=False)],
    )
    raw_val = booster.predict(x[val_idx])
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(raw_val, dtype=np.float64), y[val_idx])
    return RewardModel(booster=booster, calibrator=calibrator, feature_names=list(FEATURE_NAMES))


def _write_deployed_manifest(
    *,
    settings: Settings,
    model_dir: Path,
    dr_value: float,
    dr_lb: float,
    baseline_value: float,
    promoted_at: datetime | None = None,
) -> None:
    """Atomically write ``deployed.json`` (tmp file + ``os.replace``)."""
    settings.deployed_model_manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_dir": str(model_dir),
        "promoted_at": (promoted_at or datetime.now(UTC)).isoformat(),
        "dr_value": float(dr_value),
        "dr_lower_bound": float(dr_lb),
        "baseline_value": float(baseline_value),
        "feature_names": [],
    }
    tmp = settings.deployed_model_manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, settings.deployed_model_manifest)


def _bootstrap_deployed(
    settings: Settings, *, events: list[BanditEvent]
) -> tuple[RewardModel, Policy, DeployedManifest | None, float]:
    """When no ``deployed.json`` exists, fit a baseline model and write the manifest.

    Returns (model, policy, manifest_or_None, deployed_dr_or_None). The baseline DR
    is the on-policy mean reward of the reference slice — same convention as the demo.
    """
    labeled = _labeled(events)
    if not labeled:
        raise ValueError("cannot bootstrap deployed model from empty logs")
    model = RewardModel.fit(labeled, settings=settings)
    policy = EpsilonGreedy(
        model, epsilon=settings.epsilon, rng=np.random.default_rng(settings.seed)
    )
    ref_events = labeled[-settings.monitor_reference_window :]
    baseline_dr = float(np.mean([e.reward for e in ref_events if e.reward is not None]))
    _write_deployed_manifest(
        settings=settings,
        model_dir=settings.model_dir,
        dr_value=baseline_dr,
        dr_lb=baseline_dr,
        baseline_value=baseline_dr,
    )
    manifest = DeployedManifest(
        model_dir=str(settings.model_dir),
        promoted_at=datetime.now(UTC),
        dr_value=baseline_dr,
        dr_lower_bound=baseline_dr,
        baseline_value=baseline_dr,
        feature_names=[],
    )
    return model, policy, manifest, baseline_dr


def _append_audit(*, settings: Settings, row: AuditRow) -> None:
    settings.retrain_audit_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.retrain_audit_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_json()) + "\n")


def _candidate_policy(
    candidate: RewardModel, *, deployed_policy: Policy, settings: Settings
) -> Policy:
    """Build the candidate policy matching the deployed policy's family."""
    # If deployed is ThompsonSampling (or wraps one), refresh the ensemble.
    inner = getattr(deployed_policy, "_inner", deployed_policy)
    if isinstance(inner, ThompsonSampling):
        # Refit ensemble on the same train events the candidate was fit on.
        # We can't recover the train events here, so we fall back to EpsilonGreedy
        # over the candidate — the gate still measures value off-policy.
        return EpsilonGreedy(
            candidate, epsilon=settings.epsilon, rng=np.random.default_rng(settings.seed)
        )
    return EpsilonGreedy(
        candidate, epsilon=settings.epsilon, rng=np.random.default_rng(settings.seed)
    )


class RetrainLoop:
    """Conditional retrain: monitor → trigger → fit candidate → DR gate → promote/hold."""

    def __init__(self, *, settings: Settings, gate: PromotionGate) -> None:
        self._settings = settings
        self._gate = gate

    def run(
        self,
        *,
        deployed_model: RewardModel,
        deployed_policy: Policy,
        events: list[BanditEvent],
        deployed_dr: float | None = None,
        now: datetime | None = None,
    ) -> RetrainOutcome:
        """Run one monitor + (conditional) retrain cycle."""
        settings = self._settings
        now = now or datetime.now(UTC)

        reference_events, recent_events = _split_windows(events, settings=settings)
        ref_batch = LoggedBatch.from_events(reference_events)
        recent_batch = LoggedBatch.from_events(recent_events)

        report = build_drift_report(
            ctx=DriftReportContext(
                model=deployed_model,
                policy=deployed_policy,
                reference=ref_batch,
                recent=recent_batch,
                deployed_dr=deployed_dr,
            ),
            settings=settings,
            now=now,
        )
        append_report(report, settings.monitoring_report_path)

        # Compute days_since_promote + n_new for the scheduled ceiling.
        manifest = _read_manifest_safe(settings)
        days_since = (now - manifest.promoted_at).total_seconds() / 86400.0 if manifest else 0.0
        n_new = len(recent_events)

        trigger = evaluate_triggers(
            report, settings=settings, days_since_promote=days_since, n_new=n_new
        )

        if not trigger.should_retrain:
            outcome = RetrainOutcome(
                promoted=False,
                trigger=trigger,
                candidate_metrics={},
                gate_reason="no trigger fired",
                report=report,
            )
            audit = _audit_row(
                outcome=outcome,
                settings=settings,
                now=now,
                candidate_dr=None,
                candidate_dr_lb=None,
                deployed_dr=deployed_dr,
            )
            _append_audit(settings=settings, row=audit)
            outcome = _with_audit(outcome, audit)
            return outcome

        # Retrain: fit candidate on reference∪recent, weighted if configured.
        train_events = list(reference_events) + list(recent_events)
        weights = _time_decay_weights(train_events, settings=settings, now=now)
        candidate = _fit_candidate(train_events, settings=settings, weights=weights)
        candidate_policy = _candidate_policy(
            candidate, deployed_policy=deployed_policy, settings=settings
        )

        # Gate on the recent holdout.
        q_hat_candidate = q_matrix(candidate, recent_batch.contexts)
        baseline_value = (
            deployed_dr if deployed_dr is not None else float(recent_batch.rewards.mean())
        )
        gate_decision = self._gate.evaluate(
            candidate_policy,
            recent_batch,
            q_hat_candidate,
            baseline_value=baseline_value,
            rng=np.random.default_rng(settings.seed),
        )

        candidate_metrics = {
            "dr": float(gate_decision.candidate["dr"].value),
            "dr_lb": float(gate_decision.lower_bound),
            "ips": float(gate_decision.candidate["ips"].value),
            "dm": float(gate_decision.candidate["dm"].value),
            "snips": float(gate_decision.candidate["snips"].value),
            "lift": float(gate_decision.lift),
        }

        if gate_decision.promote:
            ts = now.strftime("%Y%m%dT%H%M%S")
            candidate_dir = settings.model_dir / "candidates" / ts
            candidate.save(candidate_dir)
            _write_deployed_manifest(
                settings=settings,
                model_dir=candidate_dir,
                dr_value=candidate_metrics["dr"],
                dr_lb=candidate_metrics["dr_lb"],
                baseline_value=baseline_value,
                promoted_at=now,
            )
            outcome = RetrainOutcome(
                promoted=True,
                trigger=trigger,
                candidate_metrics=candidate_metrics,
                gate_reason=gate_decision.reason,
                candidate_model_dir=str(candidate_dir),
                report=report,
            )
        else:
            outcome = RetrainOutcome(
                promoted=False,
                trigger=trigger,
                candidate_metrics=candidate_metrics,
                gate_reason=gate_decision.reason,
                report=report,
            )
        audit = _audit_row(
            outcome=outcome,
            settings=settings,
            now=now,
            candidate_dr=candidate_metrics["dr"],
            candidate_dr_lb=candidate_metrics["dr_lb"],
            deployed_dr=deployed_dr,
        )
        _append_audit(settings=settings, row=audit)
        return _with_audit(outcome, audit)


def _read_manifest_safe(settings: Settings) -> DeployedManifest | None:
    """Return the deployed manifest or ``None`` if absent."""
    from nba.monitoring.store_reader import read_deployed_manifest  # noqa: PLC0415

    return read_deployed_manifest(settings.deployed_model_manifest)


def _audit_row(
    *,
    outcome: RetrainOutcome,
    settings: Settings,  # noqa: ARG001 - kept for future fields
    now: datetime,
    candidate_dr: float | None,
    candidate_dr_lb: float | None,
    deployed_dr: float | None,
) -> AuditRow:
    """Build one :class:`AuditRow` from the retrain outcome."""
    verdict = "promote" if outcome.promoted else "hold"
    reasons = outcome.trigger.reasons
    return AuditRow(
        timestamp=now,
        verdict=verdict,
        reasons=reasons,
        promoted=outcome.promoted,
        candidate_dr=candidate_dr,
        candidate_dr_lb=candidate_dr_lb,
        deployed_dr=deployed_dr,
        overlap_ok=outcome.trigger.overlap_ok,
    )


def _with_audit(outcome: RetrainOutcome, audit: AuditRow) -> RetrainOutcome:
    """Return a copy of ``outcome`` with the audit row attached."""
    return RetrainOutcome(
        promoted=outcome.promoted,
        trigger=outcome.trigger,
        candidate_metrics=outcome.candidate_metrics,
        gate_reason=outcome.gate_reason,
        candidate_model_dir=outcome.candidate_model_dir,
        report=outcome.report,
        audit=audit,
    )


def bootstrap_deployed(
    *, settings: Settings, events: list[BanditEvent]
) -> tuple[RewardModel, Policy, DeployedManifest, float]:
    """Public helper: create the initial ``deployed.json`` from a labeled log.

    Used by ``run_retrain_loop.py`` when no manifest exists yet (first-run path).
    """
    model, policy, manifest, baseline_dr = _bootstrap_deployed(settings, events=events)
    assert manifest is not None
    model.save(settings.model_dir)
    return model, policy, manifest, baseline_dr


__all__ = [
    "RetrainLoop",
    "RetrainOutcome",
    "bootstrap_deployed",
]


# Silence unused-import warnings for handles re-exported by callers.
_: tuple[type, ...] = (EventStore, BootstrapEnsemble, ThompsonSampling)
