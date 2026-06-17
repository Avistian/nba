# Routing — Rules

## R1: Distance matrices are travel times, not raw km

`DistanceEngine.time_matrix` returns seconds. Routing and orchestrator assume time units.

*Confirmed: `distance.py`, orchestrator `plan_route`.*

## R2: Price doors with orchestrator.door_profit (bandit-weighted default)

Router profit input should reflect policy exploration unless explicitly comparing argmax pricing.

*Confirmed: `orchestrator.py`, Phase 7 tests, demo notebook.*

## R3: TSP-P must allow dropping optional nodes

Use `AddDisjunction` drop penalties, not forced full visits. Far low-profit doors should appear in
`route.dropped`.

*Confirmed: `test_e2e.py::test_router_drops_far_outliers`, routing tests.*

## R4: Swap distance backend via protocol, not caller changes

New engines implement `DistanceEngine`; orchestrator and solver stay unchanged.

*Confirmed: `OSRMEngine` stub pattern in `distance.py`.*

## R5: Dense geography for walkable demos/tests

When testing routing behavior, use tight clusters or `_dense_block` — raw simulator scatter yields
all-dropped routes.

*Confirmed: Phase 8 demo debugging (all 24 doors dropped before dense_block fix).*
