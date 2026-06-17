# Routing — Hypotheses

## H1: OSRM real road times materially change drop/visit decisions vs Haversine

Great-circle underestimates detours around obstacles. Hypothesis: 10–20% of dropped/visited sets flip
on a real neighborhood OSRM matrix.

*Evidence: OSRM stub only; 0 comparison runs.*

## H2: Territory clustering before TSP-P improves solve time without hurting profit

KMeans territories keep OR-Tools instances small. Hypothesis: profit capture ≥ 95% of monolithic
solve at 10× speed on 500+ doors.

*Evidence: territories module exists; not wired into main demo loop yet.*

## H3: Replan every N doors (N≈10) improves realized reward vs static route

Demo replans on remaining doors. Hypothesis: benefit scales with model drift within shift; negligible
with fixed policy + stationary world.

*Evidence: demo uses replan_every=10; no A/B vs static route logged.*
