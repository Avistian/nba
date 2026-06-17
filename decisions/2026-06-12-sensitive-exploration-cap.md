## Decision: Cap exploration in sensitive contexts via EthicalPolicy wrapper

## Context

Doors with many prior contacts are flagged sensitive (harassment risk). Phase 8 needed behavioral
ethics beyond the feature allow-list, without breaking OPE overlap.

## Alternatives considered

1. **Hard ban on non-greedy actions in sensitive contexts** — breaks full support (kills OPE).
2. **Cap explore mass to ceiling (default 5%)** — shrink distribution toward mode, all p > 0.
3. **No behavioral cap** — allow-list only.

## Reasoning

Option 2 reduces experimentation on sensitive doors while preserving IPS/DR validity. Wrapper
pattern keeps base policies unchanged. Flag uses `prior_interactions`, not demographics.

## Trade-offs accepted

- Threshold (4 prior contacts) and ceiling (0.05) are defaults needing field validation.
- Sensitive flag rate not yet analyzed on full log distribution.

## Supersedes

(none)
