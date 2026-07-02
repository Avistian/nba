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


# --- Phase 10: orienteering (budget), team routing, back-compat ---------------------------------


def test_num_vehicles_one_returns_single_route_equal_to_default() -> None:
    coords = [(42.0, -93.6)] + [(42.0 + 0.0005 * k, -93.6) for k in range(4)]
    profits = [0.0] + [1.0] * 4
    tm = _tm(coords)

    explicit = solve_tsp_profits(coords, profits, tm, num_vehicles=1, time_limit_s=2.0)
    default = solve_tsp_profits(coords, profits, tm, time_limit_s=2.0)

    assert isinstance(explicit, Route)  # single rep => single Route, never a list
    assert explicit == default


def test_time_budget_bounds_route_and_tighter_drops_more() -> None:
    # A line of eight equally-spaced high-profit doors; only the budget limits how many are served.
    coords = [(42.0, -93.6)] + [(42.0, -93.6 + 0.002 * k) for k in range(1, 9)]
    profits = [0.0] + [1.0] * 8
    tm = _tm(coords)

    generous = solve_tsp_profits(coords, profits, tm, route_budget_s=8000.0, time_limit_s=2.0)
    tight = solve_tsp_profits(coords, profits, tm, route_budget_s=1500.0, time_limit_s=2.0)
    assert isinstance(generous, Route)
    assert isinstance(tight, Route)

    # Every route stays within its budget (tolerance covers integer travel-time rounding).
    assert generous.total_time_s <= 8000.0 + len(generous.order)
    assert tight.total_time_s <= 1500.0 + len(tight.order)
    # A tighter budget can only drop the same or more doors.
    assert len(tight.visited) <= len(generous.visited)
    assert len(tight.dropped) >= len(generous.dropped)


def test_time_budget_composes_with_seconds_of_day_windows() -> None:
    # Windows put the clock in seconds-of-day (16:00-19:00); the budget bounds the shift *duration*,
    # so the two must not conflict into infeasibility (regression guard for the span-bound fix).
    coords = [(42.0, -93.6)] + [(42.0, -93.6 + 0.0008 * k) for k in range(1, 7)]
    profits = [0.0] + [1.0] * 6
    tm = _tm(coords)
    windows = [(16 * 3600, 19 * 3600)] * len(coords)

    tight = solve_tsp_profits(
        coords, profits, tm, time_windows=windows, route_budget_s=1200.0, time_limit_s=2.0
    )
    loose = solve_tsp_profits(
        coords, profits, tm, time_windows=windows, route_budget_s=3.0 * 3600, time_limit_s=2.0
    )
    assert isinstance(tight, Route)
    assert isinstance(loose, Route)
    # A 20-minute shift can service fewer doors than a 3-hour one, but both stay feasible.
    assert len(tight.visited) <= len(loose.visited)


def test_team_routing_partitions_doors_without_double_serving() -> None:
    coords = [(42.0, -93.6)] + [(42.0 + 0.0005 * k, -93.6 + 0.0005 * k) for k in range(8)]
    profits = [0.0] + [1.0] * 8
    tm = _tm(coords)

    routes = solve_tsp_profits(coords, profits, tm, num_vehicles=2, time_limit_s=2.0)

    assert isinstance(routes, list)
    assert len(routes) == 2

    all_visited = [node for r in routes for node in r.visited]
    assert len(all_visited) == len(set(all_visited))  # no door served by two reps

    served = set(all_visited)
    dropped = set(routes[0].dropped)  # dropped is the global no-rep-served set
    assert served.isdisjoint(dropped)
    assert served | dropped == set(range(1, 9))
    for r in routes:  # each rep's route is an independent depot-to-depot walk
        assert r.order[0] == 0
        assert r.order[-1] == 0


def test_per_rep_start_end_depots() -> None:
    # Two distinct depots (indices 0, 1) and four doors (indices 2..5).
    coords = [(42.0, -93.60), (42.0, -93.70)]
    coords += [(42.0 + 0.0005 * k, -93.65) for k in range(4)]
    profits = [0.0, 0.0] + [1.0] * 4
    tm = _tm(coords)

    routes = solve_tsp_profits(
        coords, profits, tm, num_vehicles=2, starts=[0, 1], ends=[0, 1], time_limit_s=2.0
    )

    assert isinstance(routes, list)
    assert len(routes) == 2
    assert routes[0].order[0] == 0 and routes[0].order[-1] == 0
    assert routes[1].order[0] == 1 and routes[1].order[-1] == 1
    served = {node for r in routes for node in r.visited}
    assert served <= {2, 3, 4, 5}  # only doors are serviceable, never a depot


def test_invalid_team_and_budget_params_raise() -> None:
    coords = [(42.0, -93.600), (42.0, -93.601)]
    tm = _tm(coords)
    profits = [0.0, 1.0]

    with pytest.raises(ValueError):
        solve_tsp_profits(coords, profits, tm, num_vehicles=0)
    with pytest.raises(ValueError):
        solve_tsp_profits(coords, profits, tm, starts=[0])  # ends missing
    with pytest.raises(ValueError):
        solve_tsp_profits(coords, profits, tm, num_vehicles=2, starts=[0], ends=[0])  # wrong length
    with pytest.raises(ValueError):
        solve_tsp_profits(coords, profits, tm, route_budget_s=-1.0)
