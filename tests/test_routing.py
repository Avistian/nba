"""Tests for the TSP-with-Profits solver."""

from __future__ import annotations

import numpy as np
import pytest

from nba.routing.distance import HaversineEngine
from nba.routing.tsp_profits import Route, solve_tsp_profits

_ENGINE = HaversineEngine(speed_kmh=4.5)


def _tm(coords: list[tuple[float, float]]) -> np.ndarray:
    return _ENGINE.time_matrix(coords)


def test_drops_far_low_profit_outlier() -> None:
    # Depot + a tight high-profit cluster + one far, low-profit door.
    coords = [(42.000, -93.600)]
    coords += [(42.000 + 0.0005 * k, -93.600) for k in range(5)]
    coords.append((42.200, -93.600))  # ~22 km away
    profits = [0.0] + [1.0] * 5 + [0.01]

    route = solve_tsp_profits(coords, profits, _tm(coords), time_limit_s=2.0)

    assert 6 in route.dropped
    assert set(range(1, 6)).issubset(set(route.visited))


def test_raising_profit_moves_door_from_dropped_to_visited() -> None:
    coords = [(42.0, -93.600), (42.0, -93.6005), (42.2, -93.600)]  # depot, near, far
    tm = _tm(coords)

    low = solve_tsp_profits(coords, [0.0, 1.0, 0.001], tm, time_limit_s=2.0)
    assert 2 in low.dropped

    high = solve_tsp_profits(coords, [0.0, 1.0, 100.0], tm, time_limit_s=2.0)
    assert 2 in high.visited


def test_capacity_caps_visited_count() -> None:
    coords = [(42.0, -93.6)] + [(42.0 + 0.0005 * k, -93.6) for k in range(8)]
    profits = [0.0] + [1.0] * 8

    route = solve_tsp_profits(coords, profits, _tm(coords), capacity=3, time_limit_s=2.0)

    assert len(route.visited) <= 3


def test_impossible_time_window_drops_door() -> None:
    # All three doors are close and high-profit; only door 2's window forces the drop.
    coords = [(42.0, -93.600), (42.0, -93.601), (42.0, -93.602)]
    profits = [0.0, 1.0, 1.0]
    # Door 2 must be reached within 1 second of the day, but travel takes far longer.
    time_windows = [(0, 100_000), (0, 100_000), (0, 1)]

    route = solve_tsp_profits(
        coords, profits, _tm(coords), time_windows=time_windows, time_limit_s=2.0
    )

    assert 2 in route.dropped
    assert 1 in route.visited


def test_generous_time_windows_visit_all() -> None:
    coords = [(42.0, -93.600), (42.0, -93.601), (42.0, -93.602)]
    profits = [0.0, 1.0, 1.0]
    time_windows = [(0, 100_000)] * 3

    route = solve_tsp_profits(
        coords, profits, _tm(coords), time_windows=time_windows, time_limit_s=2.0
    )

    assert set(route.visited) == {1, 2}


def test_route_is_deterministic() -> None:
    coords = [(42.0, -93.6)] + [(42.0 + 0.001 * k, -93.6 + 0.0005 * k) for k in range(6)]
    profits = [0.0] + [1.0, 0.5, 1.0, 0.2, 0.8, 1.0]
    tm = _tm(coords)

    r1 = solve_tsp_profits(coords, profits, tm, time_limit_s=2.0)
    r2 = solve_tsp_profits(coords, profits, tm, time_limit_s=2.0)

    assert r1 == r2


def test_total_profit_sums_visited_doors() -> None:
    coords = [(42.0, -93.6)] + [(42.0 + 0.0005 * k, -93.6) for k in range(4)]
    profits = [0.0, 1.0, 0.7, 0.5, 0.9]
    tm = _tm(coords)

    route = solve_tsp_profits(coords, profits, tm, time_limit_s=2.0)

    expected = sum(profits[v] for v in route.visited)
    assert route.total_profit == pytest.approx(expected)
    assert route.total_time_s >= 0.0


def test_single_door_is_visited_when_profitable() -> None:
    coords = [(42.0, -93.600), (42.0, -93.601)]
    route = solve_tsp_profits(coords, [0.0, 1.0], _tm(coords), time_limit_s=2.0)
    assert route.visited == [1]
    assert route.dropped == []


def test_empty_door_set_yields_empty_route() -> None:
    route = solve_tsp_profits([], [], np.zeros((0, 0)), time_limit_s=2.0)
    assert route == Route(order=[], visited=[], dropped=[], total_time_s=0.0, total_profit=0.0)


def test_depot_only_yields_empty_visits() -> None:
    route = solve_tsp_profits([(42.0, -93.6)], [0.0], np.zeros((1, 1)), time_limit_s=2.0)
    assert route.visited == []
    assert route.dropped == []
    assert route.order == [0]
