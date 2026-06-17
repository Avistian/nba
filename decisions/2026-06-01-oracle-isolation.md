## Decision: Learning modules must never import the simulator oracle

## Context

The D2D simulator exposes ground-truth functions (`true_reward`, `latent_scores`, etc.) for
generating data and grading evaluations. In production there is no oracle. If learning code peeks
at truth, offline metrics become meaningless.

## Alternatives considered

1. **Soft convention** — document "don't import oracle" in README only.
2. **Hard isolation + automated test** — ban imports in `reward`, `bandits`, `ope`, `routing`,
   `api`, `pipeline`; AST-scan in CI.
3. **Separate oracle package** — move simulator to optional extra not installed in serving image.

## Reasoning

Option 2 gives enforceable discipline without deployment complexity of option 3. Convention alone
(option 1) failed in every team we've seen — one "just for debugging" import poisons the loop.

## Trade-offs accepted

- Scripts/notebooks/tests may use oracle for evaluation — discipline is on humans there.
- AST scan won't catch dynamic imports (acceptable; grep reviews catch those).

## Supersedes

(none — foundational)
