## Decision: Route doors using bandit-weighted expected profit, not greedy argmax q

## Context

The orchestrator feeds per-door profit into TSP-P. Greedy `max q` ignores that the policy
explores — a door's realized value under the deployed policy is Σ π(a|x)·q(x,a).

## Alternatives considered

1. **Argmax q** — price each door by best single action (ignores exploration).
2. **Bandit-weighted Σ π(a|x)·q(x,a)** — expected value under actual policy (default).
3. **Sampled profit** — draw one action per door from π for routing (noisy, harder to test).

## Reasoning

Option 2 is the correct expectation for a stochastic policy and threads exploration into routing
economics. `argmax_profit=True` toggle kept for comparison in demo/notebook.

## Trade-offs accepted

- Slightly lower routed profit vs argmax when policy is nearly greedy (acceptable).
- Routing profit changes if policy changes even when q is fixed.

## Supersedes

(none)
