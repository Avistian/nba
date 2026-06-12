"""Tests for the travel-time engines."""

from __future__ import annotations

import numpy as np
import pytest

from nba.routing.distance import DistanceEngine, HaversineEngine, OSRMEngine

# One degree of latitude ~ 111.195 km on the WGS-84 authalic sphere, so this many degrees
# is very close to 1 km north-south.
_ONE_KM_DEG = 1.0 / 111.195


def test_haversine_matrix_is_symmetric_zero_diag_nonneg() -> None:
    eng = HaversineEngine(speed_kmh=4.5)
    coords = [(42.0, -93.60), (42.01, -93.60), (42.00, -93.62), (41.99, -93.61)]
    m = eng.time_matrix(coords)

    assert m.shape == (4, 4)
    assert np.allclose(np.diag(m), 0.0)
    assert np.allclose(m, m.T)
    assert np.all(m >= 0.0)


def test_haversine_known_distance() -> None:
    # Two points ~1 km apart, walking at 5 km/h -> 1/5 h = 720 s.
    eng = HaversineEngine(speed_kmh=5.0)
    m = eng.time_matrix([(0.0, 0.0), (_ONE_KM_DEG, 0.0)])
    assert m[0, 1] == pytest.approx(720.0, rel=0.02)


def test_haversine_rejects_nonpositive_speed() -> None:
    with pytest.raises(ValueError):
        HaversineEngine(speed_kmh=0.0)


def test_haversine_empty_coords() -> None:
    assert HaversineEngine(speed_kmh=4.5).time_matrix([]).shape == (0, 0)


def test_haversine_conforms_to_protocol() -> None:
    assert isinstance(HaversineEngine(speed_kmh=4.5), DistanceEngine)


def test_osrm_conforms_but_is_stubbed() -> None:
    eng = OSRMEngine("http://localhost:5000/")
    assert isinstance(eng, DistanceEngine)
    assert eng.base_url == "http://localhost:5000"
    with pytest.raises(NotImplementedError):
        eng.time_matrix([(0.0, 0.0), (1.0, 1.0)])
