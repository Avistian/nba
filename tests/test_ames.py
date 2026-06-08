"""Tests for :mod:`nba.data.ames`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba.config import Settings
from nba.data import ames as ames_mod
from nba.data.ames import AMES_COLUMNS, load_ames, synthetic_ames


def test_synthetic_ames_columns_and_positivity() -> None:
    frame = synthetic_ames(100, np.random.default_rng(0))
    assert len(frame) == 100
    for col in AMES_COLUMNS:
        assert col in frame.columns
    assert (frame["sale_price"] > 0).all()
    assert (frame["lot_area"] > 0).all()


def test_synthetic_ames_is_deterministic() -> None:
    a = synthetic_ames(50, np.random.default_rng(7))
    b = synthetic_ames(50, np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


def test_load_ames_falls_back_offline(settings: Settings, monkeypatch) -> None:
    def _boom() -> pd.DataFrame:
        raise OSError("network disabled")

    monkeypatch.setattr(ames_mod, "_download_ames", _boom)
    frame = load_ames(settings, n_fallback=120, seed=1)
    assert len(frame) == 120
    assert set(AMES_COLUMNS).issubset(frame.columns)
    # second call hits the cache and returns the same frame
    cached = load_ames(settings, n_fallback=999, seed=1)
    assert len(cached) == 120
