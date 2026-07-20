# Agentic Workflow — Rules

Apply by default when running agent-directed work in this repo.

## R1: Review in a fresh context, never the authoring session

The reviewer agent must see only the diff, not the transcript that produced it. Use a separate
`agent -p` invocation, a new chat, the `code-reviewer` subagent, or Bugbot on the PR.

*Source: article "run the reviewer agent in a fresh context window."*

## R2: Escalate ambiguous or product-changing decisions to the human

Auto-fix only unambiguous bugs. Anything that changes product behavior, or is ambiguous, stops
and asks. Never let a reviewer agent auto-apply a product decision.

*Source: article "escalate ambiguous, product-changing decisions to the human."*

## R3: Require end-to-end evidence before a PR is considered done

Passing `make check` (lint + type + unit tests) is necessary but not sufficient. Run the real
thing (`make demo` / the API / a scenario script) and attach an artifact (log/screenshot/output)
to the PR's Testing section.

*Source: article "force end-to-end evidence"; matches this repo's `quality/criteria.md`.*

## R4: One task → one agent → one branch → one PR

For parallel work, prefer Background/Cloud Agents (one per task) over shared local checkouts.
Use git worktrees only for local parallelism. Keep tasks independent.

*Source: article parallelization section; `dispatching-parallel-agents` skill.*

## R5: Keep the pipeline model-agnostic

Orchestration scripts take `--model`; do not hard-code a vendor. Preserves the ability to switch
to whichever model is currently best.

*Source: article "keep my whole workflow agent-agnostic."*

## R6: Every automated fix is a separate, conventional-message commit; every escalation is logged

Auditability is the price of autonomy. A reviewer must be able to reconstruct what the pipeline
did at a glance.
