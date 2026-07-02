"""Travel-time matrices behind a swappable :class:`DistanceEngine` interface.

Distances are expressed as travel **time in seconds**, not raw kilometres, so walking speed,
time windows, and service (dwell) times all compose in the same unit downstream. The default
:class:`HaversineEngine` is a vectorized great-circle approximation; :class:`OSRMEngine` is a
stub that documents the seam for dropping in a real road-network service later.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from nba.routing.tsp_profits import RoutingError

#: Mean Earth radius (km), WGS-84 authalic sphere.
_EARTH_RADIUS_KM = 6371.0088


@runtime_checkable
class DistanceEngine(Protocol):
    """Anything that turns coordinates into a travel-time matrix."""

    def time_matrix(self, coords: Sequence[tuple[float, float]]) -> np.ndarray:
        """Return an ``(n, n)`` matrix of travel **time in seconds**.

        The result must be symmetric with a zero diagonal and no negative entries.
        """
        ...


class HaversineEngine:
    """Great-circle travel time at a constant walking speed.

    Straight-line distance is a deliberate approximation: it needs no network data and is fully
    vectorized, so a few hundred doors cost one NumPy broadcast rather than a Python double loop.
    Swap in :class:`OSRMEngine` when a real foot network is available.
    """

    def __init__(self, *, speed_kmh: float) -> None:
        if speed_kmh <= 0.0:
            raise ValueError("speed_kmh must be > 0")
        self._speed_kmh = float(speed_kmh)

    def _haversine_km(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """Great-circle distance in km between two ``(lat, lon)`` points (degrees)."""
        lat1, lon1 = np.radians(a)
        lat2, lon2 = np.radians(b)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        return float(2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(min(1.0, h))))

    def time_matrix(self, coords: Sequence[tuple[float, float]]) -> np.ndarray:
        n = len(coords)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        arr = np.asarray(coords, dtype=np.float64)
        lat = np.radians(arr[:, 0])
        lon = np.radians(arr[:, 1])

        # Pairwise haversine via broadcasting: (n, 1) against (1, n).
        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]
        h = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
        )
        h = np.clip(h, 0.0, 1.0)
        km = 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))

        seconds = km / self._speed_kmh * 3600.0
        np.fill_diagonal(seconds, 0.0)
        # Floating-point noise can break exact symmetry; force it (OR-Tools and tests assume it).
        seconds = 0.5 * (seconds + seconds.T)
        return seconds


class OSRMEngine:
    """Road-network travel times from an OSRM Table service.

    ``time_matrix`` queries the OSRM ``/table`` endpoint over the foot profile::

        GET {base_url}/table/v1/foot/{lon1},{lat1};{lon2},{lat2};...?annotations=duration
        -> response_json["durations"]  # (n, n) list-of-lists, travel time in seconds

    Every consumer depends only on the :class:`DistanceEngine` protocol, so swapping this in for
    :class:`HaversineEngine` touches no callers. Network access is opt-in (default is Haversine);
    the single HTTP seam :meth:`_fetch` is monkeypatched in tests so CI stays offline.
    """

    #: HTTP timeout for the Table request, in seconds.
    _TIMEOUT_S = 10.0

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    @property
    def base_url(self) -> str:
        """The configured OSRM service root (no trailing slash)."""
        return self._base_url

    def _fetch(self, url: str) -> dict[str, Any]:
        """GET ``url`` and return the parsed JSON body. The one network seam (mocked in tests)."""
        try:
            with urllib.request.urlopen(url, timeout=self._TIMEOUT_S) as resp:  # noqa: S310
                status = getattr(resp, "status", 200)
                if status != 200:
                    raise RoutingError(f"OSRM returned HTTP {status} for {url}")
                payload = resp.read()
        except urllib.error.URLError as exc:  # transport failure, DNS, refused connection, ...
            raise RoutingError(f"OSRM request failed: {exc}") from exc
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RoutingError("OSRM returned a non-JSON body") from exc

    def time_matrix(self, coords: Sequence[tuple[float, float]]) -> np.ndarray:
        n = len(coords)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)

        # OSRM expects lon,lat pairs joined by ';'.
        waypoints = ";".join(f"{lon},{lat}" for lat, lon in coords)
        url = f"{self._base_url}/table/v1/foot/{waypoints}?annotations=duration"

        body = self._fetch(url)
        if body.get("code") != "Ok":
            raise RoutingError(f"OSRM response code {body.get('code')!r}: {body.get('message')}")
        durations = body.get("durations")
        if durations is None:
            raise RoutingError("OSRM response missing 'durations' block")

        matrix = np.asarray(durations, dtype=np.float64)
        if matrix.shape != (n, n):
            raise RoutingError(f"OSRM durations must be ({n}, {n}), got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise RoutingError("OSRM durations contain non-finite entries (unroutable pair?)")

        np.fill_diagonal(matrix, 0.0)
        # OSRM durations can be mildly asymmetric (one-ways); force the symmetry the protocol and
        # OR-Tools assume by averaging the two directions.
        matrix = 0.5 * (matrix + matrix.T)
        return matrix
