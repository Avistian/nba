"""Tests for :mod:`nba.config`."""

from __future__ import annotations

import pytest

from nba.config import Settings, get_settings


def test_defaults_load() -> None:
    s = Settings()
    assert s.seed == 7
    assert s.epsilon == pytest.approx(0.10)
    assert s.time_window == (16, 19)


def test_ensure_dirs_is_idempotent(settings: Settings) -> None:
    settings.ensure_dirs()
    settings.ensure_dirs()  # second call must not raise
    assert settings.data_dir.is_dir()
    assert settings.model_dir.is_dir()
    assert settings.db_path.parent.is_dir()
    # Phase 18 monitoring dirs
    assert settings.monitoring_report_path.parent.is_dir()
    assert settings.retrain_audit_path.parent.is_dir()
    assert settings.deployed_model_manifest.parent.is_dir()


def test_phase18_flags_default_off() -> None:
    """With no env overrides, Phase 18 flags preserve today's byte-identical serve/demo path."""
    s = Settings()
    assert s.use_drift_monitoring is False
    assert s.use_simulated_drift is False
    assert s.use_monitoring_dashboard is False
    assert s.metrics_exporter_enabled is False
    assert s.monitor_reference_window == 20_000
    assert s.monitor_recent_window == 2_000
    assert s.retrain_max_age_days == 30
    assert s.drift_reward_psi_threshold == pytest.approx(0.15)
    assert s.drift_feature_psi_threshold == pytest.approx(0.20)
    assert s.metrics_exporter_port == 9091
    assert s.retrain_time_decay_halflife_days is None
    # Phase 19 alert flags
    assert s.alert_email_enabled is False
    assert s.alert_smtp_host == ""
    assert s.alert_min_triggered_signals == 1
    assert s.alert_debounce_minutes == 30


def test_phase10_flags_default_to_today() -> None:
    """With no env overrides, Phase 10 flags reproduce single-vehicle, window-only routing."""
    s = Settings()
    assert s.use_time_budget is False
    assert s.shift_hours == pytest.approx(8.0)
    assert s.num_vehicles == 1
    assert s.vehicle_starts is None
    assert s.vehicle_ends is None
    assert s.distance_engine == "haversine"
    assert s.osrm_url == "http://localhost:5000"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBA_EPSILON", "0.3")
    get_settings.cache_clear()
    assert get_settings().epsilon == pytest.approx(0.3)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
