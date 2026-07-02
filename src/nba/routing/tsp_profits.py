"""TSP-with-Profits: a walkable route that may *skip* low-value, far-flung doors.

Classic TSP visits every node. Here every non-depot door is *optional*: dropping it pays a
penalty equal to its (scaled) profit. The solver therefore trades travel time against profit
collected, so a geographically isolated low-reward door is simply left out -- exactly the field
behaviour we want. Capacity caps doors per shift; per-node time windows model "residential
16:00-19:00"; a service time models the dwell of knocking and pitching. All constraints are
optional so each can be exercised in isolation.

The solver is deterministic for a fixed instance: a deterministic first solution
(``PATH_CHEAPEST_ARC``) refined by Guided Local Search under a fixed time limit on a single
thread converges to the same route on repeated runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


class RoutingError(RuntimeError):
    """Raised when the solver cannot find any feasible route for the given constraints."""


@dataclass(frozen=True)
class Route:
    """The outcome of a :func:`solve_tsp_profits` call.

    ``order`` is the full visiting sequence including the depot at both ends; ``visited`` and
    ``dropped`` are non-depot doors, in original-index space.
    """

    order: list[int]
    visited: list[int]
    dropped: list[int]
    total_time_s: float
    total_profit: float


def solve_tsp_profits(
    coords: Sequence[tuple[float, float]],
    profits: Sequence[float],
    time_matrix: np.ndarray,
    *,
    depot: int = 0,
    capacity: int | None = None,
    time_windows: list[tuple[int, int]] | None = None,
    service_time_s: float = 120.0,
    drop_scale: float = 1000.0,
    lambda_travel: float = 1.0,
    time_limit_s: float = 5.0,
    seed: int = 7,
    route_budget_s: float | None = None,
    num_vehicles: int = 1,
    starts: list[int] | None = None,
    ends: list[int] | None = None,
) -> Route | list[Route]:
    """Solve a TSP-with-Profits (optionally an Orienteering Problem for a team of reps).

    Args:
        coords: ``(lat, lon)`` per door; ``coords[depot]`` is the rep's start/end.
        profits: per-door value of servicing it; the depot's entry is ignored.
        time_matrix: ``(n, n)`` travel time in seconds (see :mod:`nba.routing.distance`).
        depot: index of the shared start/end node (ignored when ``starts``/``ends`` are given).
        capacity: max doors serviced per rep in the shift; ``None`` for unlimited.
        time_windows: per-node ``(open, close)`` in seconds-of-day; ``None`` to ignore.
        service_time_s: dwell per serviced door (knock + pitch).
        drop_scale: multiplies profit into the integer drop penalty (OR-Tools needs int costs).
        lambda_travel: weight on travel time in the arc cost, traded against drop penalties.
        time_limit_s: wall-clock budget for the metaheuristic.
        seed: reserved for reproducibility; the search is deterministic for a fixed instance.
        route_budget_s: Orienteering budget; bounds each vehicle's end-of-route cumulative time.
            ``None`` (default) leaves the route unbounded (a Prize-Collecting TSP).
        num_vehicles: reps to route (Team Orienteering). ``1`` (default) is a single rep.
        starts: per-rep start nodes; ``None`` => every rep shares ``depot``.
        ends: per-rep end nodes; ``None`` => every rep shares ``depot``. Must accompany ``starts``.

    Returns:
        A single :class:`Route` when ``num_vehicles == 1``; otherwise a list of one
        :class:`Route` per rep. Each ``dropped`` list is the set of doors served by *no* rep.

    Raises:
        RoutingError: if no feasible route exists under the constraints.
        ValueError: on shape/length mismatches or invalid team/budget parameters.
    """
    del seed  # Search is deterministic for a fixed instance; kept for API stability.

    if num_vehicles < 1:
        raise ValueError(f"num_vehicles must be >= 1, got {num_vehicles}")
    if route_budget_s is not None and route_budget_s < 0:
        raise ValueError(f"route_budget_s must be >= 0, got {route_budget_s}")
    if (starts is None) != (ends is None):
        raise ValueError("provide both starts and ends, or neither")

    def _finalize(routes: list[Route]) -> Route | list[Route]:
        return routes[0] if num_vehicles == 1 else routes

    n = len(coords)
    if n == 0:
        return _finalize([Route([], [], [], 0.0, 0.0) for _ in range(num_vehicles)])

    profit_arr = np.asarray(profits, dtype=np.float64)
    tm = np.asarray(time_matrix, dtype=np.float64)
    if profit_arr.shape != (n,):
        raise ValueError(f"profits must have length {n}, got {profit_arr.shape}")
    if tm.shape != (n, n):
        raise ValueError(f"time_matrix must be ({n}, {n}), got {tm.shape}")
    if time_windows is not None and len(time_windows) != n:
        raise ValueError(f"time_windows must have length {n}, got {len(time_windows)}")
    if not 0 <= depot < n:
        raise ValueError(f"depot {depot} out of range for {n} nodes")
    if starts is not None:
        assert ends is not None  # guaranteed by the paired-None check above
        if len(starts) != num_vehicles or len(ends) != num_vehicles:
            raise ValueError(f"starts/ends must each have length num_vehicles={num_vehicles}")
        for idx in (*starts, *ends):
            if not 0 <= idx < n:
                raise ValueError(f"vehicle depot {idx} out of range for {n} nodes")

    # Nodes that are depots (not serviceable doors): the shared depot, or every per-rep endpoint.
    depot_nodes = set(starts) | set(ends) if starts is not None and ends is not None else {depot}

    # Trivial instance: depot only, nothing to service.
    if n == 1:
        return _finalize([Route([depot], [], [], 0.0, 0.0) for _ in range(num_vehicles)])

    if starts is not None:
        assert ends is not None
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, list(starts), list(ends))
    else:
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def arc_cost(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(round(lambda_travel * float(tm[i][j])))

    cost_cb = routing.RegisterTransitCallback(arc_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_cb)

    # Every non-depot door is optional; dropping it costs its scaled profit. A single disjunction
    # per door means at most one rep may serve it (no double-serving in the team case).
    for node in range(n):
        if node in depot_nodes:
            continue
        penalty = int(round(max(0.0, float(profit_arr[node])) * drop_scale))
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    if capacity is not None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")

        def demand_cb(from_index: int) -> int:
            return 0 if manager.IndexToNode(from_index) in depot_nodes else 1

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0, [int(capacity)] * num_vehicles, True, "Capacity"
        )

    # A Time dimension is needed for per-node windows and/or the orienteering budget.
    if time_windows is not None or route_budget_s is not None:

        def time_cb(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            service = 0.0 if i in depot_nodes else service_time_s
            return int(round(float(tm[i][j]) + service))

        time_idx = routing.RegisterTransitCallback(time_cb)

        if time_windows is not None:
            horizon = max(int(close) for _, close in time_windows)
            if route_budget_s is not None:
                horizon = max(horizon, int(np.ceil(route_budget_s)))
            # slack_max = horizon allows waiting for a window to open; do not pin start to 0.
            routing.AddDimension(time_idx, horizon, horizon, False, "Time")
            time_dim = routing.GetDimensionOrDie("Time")
            for node, (open_s, close_s) in enumerate(time_windows):
                time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(int(open_s), int(close_s))
        else:
            # Budget only: no windows, so forbid slack (waiting) and start the clock at zero.
            horizon = int(np.ceil(route_budget_s)) if route_budget_s is not None else 0
            routing.AddDimension(time_idx, 0, horizon, True, "Time")
            time_dim = routing.GetDimensionOrDie("Time")

        if route_budget_s is not None:
            # Bound each rep's shift *duration* (end - start), not the absolute end cumul: with
            # seconds-of-day time windows the clock does not start at zero, so a span bound is the
            # frame-independent way to express the orienteering budget (and reduces to end <= B when
            # the clock starts at 0, i.e. the no-windows case).
            for v in range(num_vehicles):
                time_dim.SetSpanUpperBoundForVehicle(int(route_budget_s), v)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    params.log_search = False

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RoutingError(
            "no feasible route found; relax capacity/time windows/budget or check the time matrix"
        )

    def _decode(vehicle: int) -> tuple[list[int], list[int]]:
        order: list[int] = []
        index = routing.Start(vehicle)
        while not routing.IsEnd(index):
            order.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        order.append(manager.IndexToNode(index))  # closing depot
        visited = [node for node in order if node not in depot_nodes]
        return order, visited

    decoded = [_decode(v) for v in range(num_vehicles)]
    served = {node for _, visited in decoded for node in visited}
    dropped = [node for node in range(n) if node not in depot_nodes and node not in served]

    routes: list[Route] = []
    for order, visited in decoded:
        travel_s = sum(float(tm[a][b]) for a, b in zip(order[:-1], order[1:], strict=True))
        total_time_s = travel_s + service_time_s * len(visited)
        total_profit = float(sum(profit_arr[node] for node in visited))
        routes.append(
            Route(
                order=order,
                visited=visited,
                dropped=dropped,
                total_time_s=total_time_s,
                total_profit=total_profit,
            )
        )

    return _finalize(routes)
