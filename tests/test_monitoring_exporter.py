"""Tests for the Phase 18 Prometheus exporter."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from nba.config import Settings
from nba.monitoring.exporter import build_snapshot, render_prometheus_text
from nba.monitoring.signals import DriftReport, DriftSignal, append_report
from nba.monitoring.store_reader import AuditRow

_ORACLE_MODULE_PREFIXES = ("nba.data.sim", "nba.data.relational_sim", "nba.data.drift")
_ORACLE_NAMES = {
    "true_reward",
    "latent_scores",
    "true_best_action",
    "outcome_probs",
    "apply_drift_to_latent",
}


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        monitoring_report_path=tmp_path / "monitoring" / "drift_reports.jsonl",
        retrain_audit_path=tmp_path / "monitoring" / "retrain_audit.jsonl",
        deployed_model_manifest=tmp_path / "models" / "deployed.json",
    )


def _drift_report(
    *,
    reward_psi: float = 0.05,
    calib: float = 0.0,
    feature: float = 0.05,
    overlap: bool = True,
    dr_drop: float = 0.0,
) -> DriftReport:
    signals = (
        DriftSignal("reward_psi", reward_psi, 0.15, reward_psi > 0.15, f"reward PSI={reward_psi}"),
        DriftSignal("calibration_drift", calib, 0.05, calib > 0.05, f"calib Δ={calib}"),
        DriftSignal("feature_psi_max", feature, 0.20, feature > 0.20, f"max PSI={feature}"),
        DriftSignal(
            "overlap_health",
            0.5,
            0.02,
            not overlap,
            "min_p=0.5000 (floor 0.020) | ess/n=0.5000",
        ),
        DriftSignal("rolling_dr_drop", dr_drop, 0.03, dr_drop > 0.03, f"drop={dr_drop}"),
    )
    return DriftReport(
        timestamp=datetime.now(UTC),
        n_reference=2000,
        n_recent=500,
        signals=signals,
        overlap_ok=overlap,
    )


def test_render_includes_help_and_type(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)
    append_report(_drift_report(reward_psi=0.25), settings.monitoring_report_path)
    snapshot = build_snapshot(settings=settings, store=None)
    text = render_prometheus_text(snapshot, settings=settings)
    assert "# HELP nba_drift_reward_psi " in text
    assert "# TYPE nba_drift_reward_psi gauge" in text
    assert "nba_drift_reward_psi " in text


def test_threshold_companion_metrics_emitted(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)
    append_report(_drift_report(reward_psi=0.25), settings.monitoring_report_path)
    snapshot = build_snapshot(settings=settings, store=None)
    text = render_prometheus_text(snapshot, settings=settings)
    assert "nba_drift_reward_psi_threshold " in text
    assert "nba_drift_feature_psi_max_threshold " in text
    assert "nba_drift_rolling_dr_drop_threshold " in text


def test_triggered_flag_with_signal_label(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)
    append_report(_drift_report(reward_psi=0.25), settings.monitoring_report_path)
    snapshot = build_snapshot(settings=settings, store=None)
    text = render_prometheus_text(snapshot, settings=settings)
    # The triggered flag uses a signal="reward_psi" label.
    assert 'nba_drift_signal_triggered{signal="reward_psi"} 1.0' in text
    assert 'nba_drift_signal_triggered{signal="feature_psi_max"} 0.0' in text


def test_retrain_audit_counter_by_verdict(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.retrain_audit_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    row1 = AuditRow(
        timestamp=now,
        verdict="promote",
        reasons=("reward_psi",),
        promoted=True,
        candidate_dr=0.5,
        candidate_dr_lb=0.4,
        deployed_dr=0.3,
        overlap_ok=True,
    )
    settings.retrain_audit_path.write_text(
        json.dumps(row1.to_json()) + "\n",
        encoding="utf-8",
    )
    snapshot = build_snapshot(settings=settings, store=None)
    text = render_prometheus_text(snapshot, settings=settings)
    assert 'nba_retrain_total{verdict="promote"} 1.0' in text
    # hold always emitted (0 when no hold rows) so Grafana stat panels never show "No data"
    assert 'nba_retrain_total{verdict="hold"} 0.0' in text


def test_deployed_manifest_metrics(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.deployed_model_manifest.parent.mkdir(parents=True, exist_ok=True)
    settings.deployed_model_manifest.write_text(
        json.dumps(
            {
                "model_dir": str(settings.model_dir),
                "promoted_at": (datetime.now(UTC) - timedelta(days=5)).isoformat(),
                "dr_value": 0.4,
                "dr_lower_bound": 0.35,
                "baseline_value": 0.3,
                "feature_names": [],
            }
        ),
        encoding="utf-8",
    )
    snapshot = build_snapshot(settings=settings, store=None)
    text = render_prometheus_text(snapshot, settings=settings)
    assert "nba_deployed_dr_lb " in text
    assert "nba_deployed_model_age_days " in text


def test_snapshot_handles_missing_artifacts(tmp_path: Path) -> None:
    """build_snapshot must not raise when no artifacts exist yet."""
    settings = _settings(tmp_path)
    snapshot = build_snapshot(settings=settings, store=None)
    assert snapshot.report is None
    assert snapshot.deployed is None
    text = render_prometheus_text(snapshot, settings=settings)
    # With no artifacts at all, only the threshold companion metrics render (those are
    # gated on the report existing). Assert it's valid (possibly empty) text.
    assert isinstance(text, str)


def test_overlap_min_p_from_report_when_no_store(tmp_path: Path) -> None:
    """overlap_min_propensity falls back to parsing the drift report when EventStore is absent."""
    settings = _settings(tmp_path)
    settings.monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)
    append_report(_drift_report(overlap=False), settings.monitoring_report_path)
    snapshot = build_snapshot(settings=settings, store=None)
    text = render_prometheus_text(snapshot, settings=settings)
    assert "nba_drift_overlap_min_propensity " in text
    """exporter.py must never import simulator oracle modules or symbols."""
    path = Path(__file__).resolve().parents[1] / "src" / "nba" / "monitoring" / "exporter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            _ORACLE_MODULE_PREFIXES
        ):
            imported = {alias.name for alias in node.names}
            assert not (imported & _ORACLE_NAMES), f"exporter imports oracle: {imported}"
        if isinstance(node, ast.Name) and node.id in _ORACLE_NAMES:
            raise AssertionError(f"exporter references oracle symbol {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr in _ORACLE_NAMES:
            raise AssertionError(f"exporter references oracle attribute {node.attr!r}")


def test_metrics_handler_caches_across_requests(tmp_path: Path, monkeypatch) -> None:
    """Each HTTP request gets a new handler; cache must live on the handler class."""
    import importlib.util
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    settings = _settings(tmp_path)
    settings.metrics_refresh_seconds = 60

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_metrics_exporter.py"
    spec = importlib.util.spec_from_file_location("run_metrics_exporter", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    render_calls = 0

    def counting_render(_settings: Settings) -> str:
        nonlocal render_calls
        render_calls += 1
        return "# cached metrics\n"

    monkeypatch.setattr(mod, "_render", counting_render)

    handler_cls = mod._make_handler(settings)
    handler_cls._cached = ""
    handler_cls._last_refresh = 0.0

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for _ in range(2):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
                assert resp.read()
        assert render_calls == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_exporter_disabled_via_env_noop(tmp_path: Path, monkeypatch) -> None:
    """``run_metrics_exporter.py --once`` with exporter disabled exits 0 cleanly."""
    tmp_path.parents[-1] if tmp_path.parents else Path.cwd()
    env = {
        **__import__("os").environ,
        "NBA_METRICS_EXPORTER_ENABLED": "0",
        "NBA_DATA_DIR": str(tmp_path / "data"),
        "NBA_MODEL_DIR": str(tmp_path / "models"),
        "NBA_DB_PATH": str(tmp_path / "events.db"),
        "NBA_MONITORING_REPORT_PATH": str(tmp_path / "drift.jsonl"),
        "NBA_RETRAIN_AUDIT_PATH": str(tmp_path / "audit.jsonl"),
        "NBA_DEPLOYED_MODEL_MANIFEST": str(tmp_path / "deployed.json"),
        "NBA_USE_DRIFT_MONITORING": "0",
    }
    result = subprocess.run(
        [sys.executable, "scripts/run_metrics_exporter.py", "--once"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "nothing to emit" in result.stdout.lower()
