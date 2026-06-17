# Routing — Knowledge

## Problem: TSP-with-Profits (TSP-P)

Not "visit all doors" — each door is **optional**. Skipping costs a drop penalty ≈ scaled profit.
OR-Tools (`routing/tsp_profits.py`) balances travel time vs forgone profit.

Constraints supported: depot, capacity (`shift_capacity`), per-node time windows (`time_window`),
`lambda_travel`, `drop_scale`.

## Distance engine

`DistanceEngine` protocol → travel **time** matrix (seconds).

- `HaversineEngine` — great-circle distance / walking speed (default).
- `OSRMEngine` — stub for future road network.

## Territories

`territories.py` — KMeans on equal-area-scaled lat/lon to split large door sets into walkable
neighborhoods before routing.

## Demo geography note

Simulator scatters doors over km. For walkable demos, `_dense_block` in `run_demo.py` repositions
sampled contexts onto a small disk. **lat/lon excluded from model** — repositioning changes
geography only, not reward/profit.

## Routing vs visit-all

Demo compares planned route time vs naive nearest-neighbor visit-all tour. TSP-P drops far
low-profit doors (tested with injected outliers ~33 km north).
