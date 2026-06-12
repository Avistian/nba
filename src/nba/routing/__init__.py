"""Turn per-door profits into a walkable route (the "dispose" half of the loop).

The router is deliberately decoupled from the rest of the system: it consumes only
coordinates and a per-door *profit* number (typically the reward model's best-action value).
A :class:`~nba.routing.distance.DistanceEngine` produces a travel-*time* matrix, KMeans carves
the doors into walkable territories, and an OR-Tools TSP-with-Profits solver decides which doors
are worth visiting under capacity and time-window constraints.
"""
