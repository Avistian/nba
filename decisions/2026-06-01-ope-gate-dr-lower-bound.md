## Decision: Promote policies only when DR lower confidence bound beats baseline

## Context

OPE estimates are noisy. Shipping on a point estimate risks deploying policies that looked good by
luck. Phase 5 needed a conservative gate before promotion.

## Alternatives considered

1. **Point estimate** — promote if DR value > baseline (aggressive).
2. **Lower bound** — promote if DR − z·SE > baseline + min_lift (conservative).
3. **IPS-only gate** — unbiased but extremely high variance in our overlap regime.

## Reasoning

Option 2 optimizes for the expensive failure mode (bad promotion). DR is primary estimator; IPS/DM
reported for transparency. IPS-only rejected due to weight concentration warnings at realistic ε.

## Trade-offs accepted

- More HOLD decisions at moderate data sizes (demo often holds despite competitive point DR).
- Requires honest SE estimates; bootstrap SE not yet implemented.

## Supersedes

(none)
