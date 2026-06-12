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
) -> Route:
    """Solve a single-vehicle TSP-with-Profits over ``coords``.

    Args:
        coords: ``(lat, lon)`` per door; ``coords[depot]`` is the rep's start/end.
        profits: per-door value of servicing it; the depot's entry is ignored.
        time_matrix: ``(n, n)`` travel time in seconds (see :mod:`nba.routing.distance`).
        depot: index of the start/end node.
        capacity: max doors serviced in the shift; ``None`` for unlimited.
        time_windows: per-node ``(open, close)`` in seconds-of-day; ``None`` to ignore.
        service_time_s: dwell per serviced door (knock + pitch).
        drop_scale: multiplies profit into the integer drop penalty (OR-Tools needs int costs).
        lambda_travel: weight on travel time in the arc cost, traded against drop penalties.
        time_limit_s: wall-clock budget for the metaheuristic.
        seed: reserved for reproducibility; the search is deterministic for a fixed instance.

    Returns:
        The chosen :class:`Route`.

    Raises:
        RoutingError: if no feasible route exists under the constraints.
        ValueError: on shape/length mismatches.
    """
    del seed  # Search is deterministic for a fixed instance; kept for API stability.

    n = len(coords)
    if n == 0:
        return Route(order=[], visited=[], dropped=[], total_time_s=0.0, total_profit=0.0)

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

    # Trivial instance: depot only, nothing to service.
    if n == 1:
        return Route(order=[depot], visited=[], dropped=[], total_time_s=0.0, total_profit=0.0)

    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def arc_cost(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(round(lambda_travel * float(tm[i][j])))

    cost_cb = routing.RegisterTransitCallback(arc_cost)
    routing.SetArcCostEvaluatorOfAllVehicles(cost_cb)

    # Every non-depot door is optional; dropping it costs its scaled profit.
    for node in range(n):
        if node == depot:
            continue
        penalty = int(round(max(0.0, float(profit_arr[node])) * drop_scale))
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    if capacity is not None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")

        def demand_cb(from_index: int) -> int:
            return 0 if manager.IndexToNode(from_index) == depot else 1

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0, [int(capacity)], True, "Capacity"
        )

    if time_windows is not None:
        horizon = max(int(close) for _, close in time_windows)

        def time_cb(from_index: int, to_index: int) -> int:
            i = manager.IndexToNode(from_index)
            j = manager.IndexToNode(to_index)
            service = 0.0 if i == depot else service_time_s
            return int(round(float(tm[i][j]) + service))

        time_idx = routing.RegisterTransitCallback(time_cb)
        # slack_max = horizon allows waiting for a window to open; do not pin start to 0.
        routing.AddDimension(time_idx, horizon, horizon, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        for node, (open_s, close_s) in enumerate(time_windows):
            time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(int(open_s), int(close_s))

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
            "no feasible route found; relax capacity/time windows or check the time matrix"
        )

    order: list[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    order.append(manager.IndexToNode(index))  # closing depot

    visited = [node for node in order if node != depot]
    dropped = [
        node
        for node in range(n)
        if node != depot
        and solution.Value(routing.NextVar(manager.NodeToIndex(node)))
        == manager.NodeToIndex(node)
    ]

    travel_s = sum(float(tm[a][b]) for a, b in zip(order[:-1], order[1:], strict=True))
    total_time_s = travel_s + service_time_s * len(visited)
    total_profit = float(sum(profit_arr[node] for node in visited))

    return Route(
        order=order,
        visited=visited,
        dropped=dropped,
        total_time_s=total_time_s,
        total_profit=total_profit,
    )
