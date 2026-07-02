"""Tests for the travel-time engines."""

from __future__ import annotations

import urllib.error
import urllib.request

import numpy as np
import pytest

from nba.routing.distance import DistanceEngine, HaversineEngine, OSRMEngine
from nba.routing.tsp_profits import RoutingError

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


def test_osrm_conforms_to_protocol_and_trims_url() -> None:
    eng = OSRMEngine("http://localhost:5000/")
    assert isinstance(eng, DistanceEngine)
    assert eng.base_url == "http://localhost:5000"


def test_osrm_empty_coords() -> None:
    assert OSRMEngine("http://localhost:5000").time_matrix([]).shape == (0, 0)


def test_osrm_parses_and_symmetrizes_durations(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"code": "Ok", "durations": [[0, 10, 20], [12, 0, 30], [22, 28, 0]]}
    monkeypatch.setattr(OSRMEngine, "_fetch", lambda self, url: payload)

    eng = OSRMEngine("http://localhost:5000")
    m = eng.time_matrix([(42.0, -93.60), (42.01, -93.60), (42.0, -93.62)])

    assert m.shape == (3, 3)
    assert np.allclose(np.diag(m), 0.0)
    assert np.allclose(m, m.T)  # forced symmetric from the (10,12) / (20,22) / (30,28) pairs
    assert m[0, 1] == pytest.approx(11.0)
    assert m[0, 2] == pytest.approx(21.0)


def test_osrm_builds_lonlat_table_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_fetch(self: OSRMEngine, url: str) -> dict[str, object]:
        captured["url"] = url
        return {"code": "Ok", "durations": [[0, 5], [5, 0]]}

    monkeypatch.setattr(OSRMEngine, "_fetch", fake_fetch)
    OSRMEngine("http://localhost:5000").time_matrix([(42.0, -93.6), (41.0, -92.0)])

    assert "/table/v1/foot/" in captured["url"]
    assert "-93.6,42.0" in captured["url"]  # OSRM wants lon,lat order
    assert "annotations=duration" in captured["url"]


def test_osrm_raises_on_error_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        OSRMEngine, "_fetch", lambda self, url: {"code": "NoRoute", "message": "boom"}
    )
    with pytest.raises(RoutingError):
        OSRMEngine("http://x").time_matrix([(0.0, 0.0), (1.0, 1.0)])


def test_osrm_raises_on_wrong_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = {"code": "Ok", "durations": [[0, 1]]}
    monkeypatch.setattr(OSRMEngine, "_fetch", lambda self, url: bad)
    with pytest.raises(RoutingError):
        OSRMEngine("http://x").time_matrix([(0.0, 0.0), (1.0, 1.0)])


def test_osrm_wraps_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RoutingError):
        OSRMEngine("http://localhost:5000").time_matrix([(0.0, 0.0), (1.0, 1.0)])
