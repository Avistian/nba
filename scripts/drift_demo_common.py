"""Shared helpers for drift demos (batch + online).

Isolates demo artifacts under ``artifacts/drift_demo/*`` so neither demo
mutates production telemetry. Used by ``simulate_drift_demo.py`` and
``run_online_drift_demo.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nba.api.store import EventStore
from nba.config import Settings
from nba.monitoring.retrain import _split_windows, load_deployed_stack
from nba.monitoring.signals import (
    DriftReportContext,
    append_report,
    build_drift_report,
)
from nba.ope.estimators import LoggedBatch
from nba.reward.model import RewardModel
from nba.schema import BanditEvent

# Isolated artifact tree for drift demos — never the shared production defaults.
DRIFT_DEMO_ROOT = Path("artifacts/drift_demo")
DRIFT_DEMO_DB_PATH = DRIFT_DEMO_ROOT / "events.db"
DRIFT_DEMO_MODEL_DIR = DRIFT_DEMO_ROOT / "models"
DRIFT_DEMO_DEPLOYED_MANIFEST = DRIFT_DEMO_MODEL_DIR / "deployed.json"
DRIFT_DEMO_MONITORING_REPORT = DRIFT_DEMO_ROOT / "monitoring" / "drift_reports.jsonl"
DRIFT_DEMO_RETRAIN_AUDIT = DRIFT_DEMO_ROOT / "monitoring" / "retrain_audit.jsonl"

DEFAULT_PRODUCTION_DB_PATH = Path("artifacts/events.db")
DEFAULT_PRODUCTION_MODEL_DIR = Path("artifacts/models")
DEFAULT_PRODUCTION_DEPLOYED_MANIFEST = Path("artifacts/models/deployed.json")
DEFAULT_PRODUCTION_MONITORING_REPORT = Path("artifacts/monitoring/drift_reports.jsonl")
DEFAULT_PRODUCTION_RETRAIN_AUDIT = Path("artifacts/monitoring/retrain_audit.jsonl")

# Backward-compatible aliases for tests importing from simulate_drift_demo.
_DEFAULT_PRODUCTION_DB_PATH = DEFAULT_PRODUCTION_DB_PATH
_DEFAULT_PRODUCTION_MODEL_DIR = DEFAULT_PRODUCTION_MODEL_DIR
_DEFAULT_PRODUCTION_DEPLOYED_MANIFEST = DEFAULT_PRODUCTION_DEPLOYED_MANIFEST
_DEFAULT_PRODUCTION_MONITORING_REPORT = DEFAULT_PRODUCTION_MONITORING_REPORT
_DEFAULT_PRODUCTION_RETRAIN_AUDIT = DEFAULT_PRODUCTION_RETRAIN_AUDIT
_DRIFT_DEMO_DB_PATH = DRIFT_DEMO_DB_PATH
_DRIFT_DEMO_MODEL_DIR = DRIFT_DEMO_MODEL_DIR
_DRIFT_DEMO_DEPLOYED_MANIFEST = DRIFT_DEMO_DEPLOYED_MANIFEST
_DRIFT_DEMO_MONITORING_REPORT = DRIFT_DEMO_MONITORING_REPORT
_DRIFT_DEMO_RETRAIN_AUDIT = DRIFT_DEMO_RETRAIN_AUDIT


def redirect_production_path(value: Path, *, production: Path, demo: Path) -> Path:
    """Return ``demo`` when ``value`` is the shared production default."""
    if value.resolve() == production.resolve():
        return demo
    return value


def demo_db_path(settings: Settings) -> Path:
    """Return a demo-safe db path, redirecting away from the shared production store."""
    return redirect_production_path(
        settings.db_path,
        production=DEFAULT_PRODUCTION_DB_PATH,
        demo=DRIFT_DEMO_DB_PATH,
    )


def demo_settings(settings: Settings) -> Settings:
    """Redirect production artifact defaults to the isolated drift-demo tree."""
    return settings.model_copy(
        update={
            "db_path": demo_db_path(settings),
            "model_dir": redirect_production_path(
                settings.model_dir,
                production=DEFAULT_PRODUCTION_MODEL_DIR,
                demo=DRIFT_DEMO_MODEL_DIR,
            ),
            "deployed_model_manifest": redirect_production_path(
                settings.deployed_model_manifest,
                production=DEFAULT_PRODUCTION_DEPLOYED_MANIFEST,
                demo=DRIFT_DEMO_DEPLOYED_MANIFEST,
            ),
            "monitoring_report_path": redirect_production_path(
                settings.monitoring_report_path,
                production=DEFAULT_PRODUCTION_MONITORING_REPORT,
                demo=DRIFT_DEMO_MONITORING_REPORT,
            ),
            "retrain_audit_path": redirect_production_path(
                settings.retrain_audit_path,
                production=DEFAULT_PRODUCTION_RETRAIN_AUDIT,
                demo=DRIFT_DEMO_RETRAIN_AUDIT,
            ),
        }
    )


def online_demo_settings(settings: Settings) -> Settings:
    """Demo tree + knobs tuned for the live-streaming producer."""
    return demo_settings(settings).model_copy(
        update={
            "monitor_interval_events": 1,
            "monitor_recent_window": min(settings.monitor_recent_window, 500),
            "retrain_min_new_events": 100,
        }
    )


def is_drift_demo_db(db_path: Path) -> bool:
    return db_path.resolve() == DRIFT_DEMO_DB_PATH.resolve()


def mean_regret(events, oracle) -> float:
    if not events:
        return 0.0
    regrets = []
    for e in events:
        chosen = oracle.true_reward(e.context, e.action)
        best = oracle.true_reward(e.context, oracle.true_best_action(e.context))
        regrets.append(best - chosen)
    return float(np.mean(regrets))


def calib_mae(model: RewardModel, events) -> float:
    if not events:
        return 0.0
    errs = [abs(model.q(e.context, e.action) - (e.reward or 0.0)) for e in events]
    return float(np.mean(errs))


def load_promoted_stack(
    *,
    settings: Settings,
    events: list,
) -> tuple[RewardModel, object, float]:
    """Reload model + policy after promotion so drift scoring uses an aligned stack."""
    deployed_model, deployed_policy, _manifest, baseline_dr = load_deployed_stack(
        settings=settings, events=events
    )
    return deployed_model, deployed_policy, baseline_dr


def signals_from_report(report) -> tuple[dict[str, float], bool]:
    """Extract signal values from a scored DriftReport."""
    return {s.name: s.value for s in report.signals}, report.overlap_ok


def score_drift(
    *,
    model,
    policy,
    events,
    settings,
    deployed_dr,
    promoted_at=None,
):
    """Score drift with the same reference/recent windows as ``RetrainLoop.run``."""
    reference_events, recent_events = _split_windows(
        events, settings=settings, promoted_at=promoted_at
    )
    ref = LoggedBatch.from_events(reference_events)
    recent = LoggedBatch.from_events(recent_events)
    report = build_drift_report(
        ctx=DriftReportContext(
            model=model,
            policy=policy,
            reference=ref,
            recent=recent,
            deployed_dr=deployed_dr,
        ),
        settings=settings,
    )
    append_report(report, settings.monitoring_report_path)
    return signals_from_report(report)


def ingest_events_incremental(
    settings: Settings,
    events: list[BanditEvent],
    *,
    policy_name: str = "online_drift_demo",
) -> int:
    """Append labeled events to the EventStore without wiping existing rows."""
    labeled = [e for e in events if e.reward is not None and e.outcome is not None]
    if not labeled:
        return 0
    db_path = demo_db_path(settings)
    store = EventStore(db_path)
    try:
        return store.ingest_bandit_events(labeled, policy_name=policy_name)
    finally:
        store.close()


def persist_events(settings: Settings, events: list) -> int:
    """Write labeled events to the EventStore (batch demo: resets isolated demo db first)."""
    labeled = [e for e in events if e.reward is not None and e.outcome is not None]
    if not labeled:
        return 0
    db_path = demo_db_path(settings)
    if is_drift_demo_db(db_path) and db_path.exists():
        db_path.unlink()
    store = EventStore(db_path)
    try:
        return store.ingest_bandit_events(labeled, policy_name="drift_demo")
    finally:
        store.close()


def reset_demo_tree(settings: Settings) -> None:
    """Remove isolated demo artifacts so a live run starts clean."""
    paths = [
        demo_db_path(settings),
        settings.deployed_model_manifest,
        settings.monitoring_report_path,
        settings.retrain_audit_path,
        settings.monitoring_report_path.parent / "alert_state.json",
    ]
    for path in paths:
        if path.exists():
            path.unlink()
    model_dir = settings.model_dir
    if model_dir.exists():
        import shutil  # noqa: PLC0415

        shutil.rmtree(model_dir, ignore_errors=True)


# Backward-compatible private aliases for simulate_drift_demo and tests.
_demo_db_path = demo_db_path
_demo_settings = demo_settings
_is_drift_demo_db = is_drift_demo_db
_redirect_production_path = redirect_production_path
_mean_regret = mean_regret
_calib_mae = calib_mae
_load_promoted_stack = load_promoted_stack
_signals_from_report = signals_from_report
_score_drift = score_drift
_persist_events = persist_events
