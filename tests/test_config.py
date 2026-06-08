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


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBA_EPSILON", "0.3")
    get_settings.cache_clear()
    assert get_settings().epsilon == pytest.approx(0.3)


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
