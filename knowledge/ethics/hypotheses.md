# Ethics — Hypotheses

## H1: prior_interactions threshold of 4 is the right sensitive flag

May be too low (flags many doors) or too high (misses harassment risk). Needs stakeholder review +
field policy input.

*Evidence: single default chosen in Phase 8; 0 sensitivity analysis on flag rate.*

## H2: 5% exploration ceiling in sensitive contexts is sufficient ethically without killing learning

Cap reduces explore mass to ≤0.05 while keeping support. Hypothesis: OPE still has enough overlap
on sensitive rows.

*Evidence: 1 unit test on synthetic distribution; no logged-data ESS breakdown by sensitive flag.*

## H3: Behavioral sensitive flag is preferable to demographic proxies

Using prior_interactions avoids protected attributes. Hypothesis: no need for demographic features
to achieve harassment reduction goals.

*Evidence: design rationale only; no comparative study.*
