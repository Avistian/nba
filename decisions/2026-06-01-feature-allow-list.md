## Decision: Build model features from an explicit allow-list, excluding geo and protected fields

## Context

Door-to-door NBA models risk redlining (learning from lat/lon) and fairness violations (protected
attributes). Phase 1 needed an enforceable feature contract.

## Alternatives considered

1. **Reflect all ProspectContext fields** — convenient, unsafe.
2. **Explicit ALLOWED_FEATURES allow-list** — assemble vector from named fields only.
3. **Post-hoc feature filtering** — drop columns after featurization (easy to bypass).

## Reasoning

Option 2 makes the safe path the only path. Geo used for routing only (orchestrator/TSP), never in
q(x,a). Tests assert forbidden fields absent from FEATURE_NAMES and persisted model schema.

## Trade-offs accepted

- Model cannot learn fine-grained spatial patterns from coordinates (by design).
- Aggregate spatial proxies (`block_density`, `distance_from_rep_km`) are allowed — monitor for
  proxy discrimination.

## Supersedes

(none)
