"""Tests for :mod:`nba.data.features`."""

from __future__ import annotations

import numpy as np
import pytest

from nba.data.features import (
    FEATURE_NAMES,
    WEATHER_LEVELS,
    featurize,
    featurize_batch,
    n_features,
)
from nba.schema import ACTIONS, Action, ProspectContext


def _context(**overrides: object) -> ProspectContext:
    base: dict[str, object] = {
        "address_id": "a-1",
        "lat": 42.0,
        "lon": -93.0,
        "property_value": 250_000.0,
        "roof_age_years": 12.0,
        "est_income": 85_000.0,
        "tenure_years": 4.0,
        "prior_interactions": 1,
        "hour": 18,
        "dow": 2,
        "weather": "clear",
        "block_density": 20.0,
        "neighbor_recent_conversion": True,
        "distance_from_rep_km": 0.4,
        "nearby_high_reward_density": 0.7,
    }
    base.update(overrides)
    return ProspectContext(**base)  # type: ignore[arg-type]


def test_featurize_width_matches_feature_names() -> None:
    ctx = _context()
    for action in ACTIONS:
        vec = featurize(ctx, action)
        assert vec.shape == (n_features(),)
        assert len(FEATURE_NAMES) == n_features()
        assert vec.dtype == np.float64


def test_featurize_is_deterministic() -> None:
    ctx = _context()
    a = featurize(ctx, Action.KNOCK_NOW)
    b = featurize(ctx, Action.KNOCK_NOW)
    assert np.array_equal(a, b)


def test_only_action_block_changes_with_action() -> None:
    ctx = _context()
    knock = featurize(ctx, Action.KNOCK_NOW)
    flyer = featurize(ctx, Action.LEAVE_FLYER)
    ctx_width = n_features() - len(ACTIONS)
    assert np.array_equal(knock[:ctx_width], flyer[:ctx_width])
    assert not np.array_equal(knock[ctx_width:], flyer[ctx_width:])


def test_featurize_batch_matches_per_action() -> None:
    ctx = _context()
    batch = featurize_batch(ctx)
    assert batch.shape == (len(ACTIONS), n_features())
    for i, action in enumerate(ACTIONS):
        assert np.array_equal(batch[i], featurize(ctx, action))


def test_weather_onehot_sums_to_one() -> None:
    ctx = _context(weather="rain")
    vec = featurize(ctx, Action.SKIP_DOOR)
    weather_cols = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("weather=")]
    block = vec[weather_cols]
    assert block.sum() == pytest.approx(1.0)
    assert len(weather_cols) == len(WEATHER_LEVELS)
    rain_idx = FEATURE_NAMES.index("weather=rain")
    assert vec[rain_idx] == pytest.approx(1.0)


def test_action_onehot_sums_to_one() -> None:
    ctx = _context()
    vec = featurize(ctx, Action.PITCH_SOLAR)
    action_cols = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("action=")]
    assert vec[action_cols].sum() == pytest.approx(1.0)


def test_boolean_encoded_as_float() -> None:
    on = featurize(_context(neighbor_recent_conversion=True), Action.KNOCK_NOW)
    off = featurize(_context(neighbor_recent_conversion=False), Action.KNOCK_NOW)
    idx = FEATURE_NAMES.index("neighbor_recent_conversion")
    assert on[idx] == pytest.approx(1.0)
    assert off[idx] == pytest.approx(0.0)


def test_allow_list_excludes_geo_and_identity() -> None:
    for forbidden in ("lat", "lon", "address_id"):
        assert forbidden not in FEATURE_NAMES
