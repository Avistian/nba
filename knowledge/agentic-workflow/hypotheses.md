# Agentic Workflow — Hypotheses

Need real usage data before promoting to rules.

## H1: A `stop`-hook gate that blocks (exit 2) until E2E evidence exists raises quality more than it annoys

Blocking the agent from "finishing" until it has produced an artifact could force the E2E habit.
Risk: false stops on trivial changes (docs, comments). Test by running `stop_gate.sh` in
observe-only mode for a while, counting how often it *would* have blocked correctly.

*Status: 0 confirmations.*

## H2: Bugbot alone covers step 4 (peer review), so `no_mistakes.sh` can skip its own reviewer agent

If Bugbot's findings on the PR are as good as a bespoke `agent -p` reviewer, we save a step and
tokens. Test by running both on the same diffs and comparing catch rate.

*Status: 0 confirmations.*

## H3: `gnhf`-style rollback-on-failure beats a single long agent session for phase-sized work

Fresh context per step + auto-rollback should avoid the "context fills, compacts, loses the
thread" failure the article describes. Test on the next phase: run it once via a single Cloud
Agent and once via `gnhf`, compare commit cleanliness and rework.

*Status: 0 confirmations.*

## H4: The article's "68% of changes had bugs" holds in this repo

If a fresh-context reviewer flags real issues on most agent PRs here too, that justifies making
`no-mistakes` mandatory (a quality-gate criterion). Track: of PRs run through the pipeline, what
fraction had at least one auto-fixed obvious bug or escalation.

*Status: 0 confirmations (article claim, not yet reproduced locally).*
