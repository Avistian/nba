"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from nba.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _seed_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the seed and clear the settings cache around every test."""
    monkeypatch.setenv("NBA_SEED", "7")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A tmp-path-backed Settings instance."""
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "artifacts" / "models",
        db_path=tmp_path / "artifacts" / "events.db",
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded numpy Generator."""
    return np.random.default_rng(0)
