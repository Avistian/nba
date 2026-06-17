## Decision: Verify regret as "far below random," not "decreasing curve" within a single shift

## Context

PLAN.md originally claimed "cumulative regret trends down." Phase 8 demo runs a **fixed**
already-gated policy for one shift — no within-shift learning. Per-round regret is stationary.

## Alternatives considered

1. **Assert negative slope on cumulative regret** — fails on fixed-policy shift (false negative).
2. **Assert bandit avg regret << uniform avg regret** — measures near-optimality (valid).
3. **Add online retraining in demo to get decreasing curve** — heavier, blurs demo scope.

## Reasoning

Option 2 matches what a single deployed shift can prove. Decreasing regret is an online-learning
claim for multi-shift retraining (future work). Document honestly in PLAN.md and docs/09.

## Trade-offs accepted

- Marketing language "regret trends down" needs qualification in docs.
- E2E test does not check slope; checks relative regret vs uniform.

## Supersedes

(none — clarifies PLAN wording, does not reverse a code decision)
