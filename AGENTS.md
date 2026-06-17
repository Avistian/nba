## Knowledge Architecture

Before starting a new task, review existing rules and hypotheses for this domain.
Apply rules by default. Check if any hypothesis can be tested with today's work.

At the end of each task, extract insights. Store them in domain folders, e.g.:
  /knowledge/pricing/         (or /onboarding/, /competitors/)
    knowledge.md  (facts and patterns)
    hypotheses.md (need more data)
    rules.md      (confirmed — apply by default)

Maintain a /knowledge/INDEX.md that routes to each domain folder.
Create the structure if it doesn't exist yet.
When a hypothesis gets confirmed 3+ times, promote it to a rule.
When a rule gets contradicted by new data, demote it back to hypothesis.

## Decision Journal

When about to make a decision that affects more than today's task, first grep /decisions/ for prior decisions in that area. Follow them unless new information invalidates the reasoning.

If no prior decision exists — or you're replacing one — log it:

File: /decisions/YYYY-MM-DD-{topic}.md

Format:
  ## Decision: {what you decided}
  ## Context: {why this came up}
  ## Alternatives considered: {what else was on the table}
  ## Reasoning: {why this option won}
  ## Trade-offs accepted: {what you gave up}
  ## Supersedes: {link to prior decision, if replacing}

## Quality Gate

Before marking any task complete, evaluate it against the quality criteria for this project:

File: /quality/criteria.md

Format:
  ## Category: {area — e.g., API design, UI, data}
  ## Criteria:
    - {specific, testable check}
    - {specific, testable check}
  ## Severity: blocking | warning
  ## Source: {where this criterion came from}
  ## Last triggered: {date, or "never"}

If /quality/criteria.md doesn't exist, create it with initial criteria based on the project's domain and standards. Ask the user to review.

After evaluation, update criteria:
  - Criteria that caught a real issue: note the date
  - Criteria triggered 3+ times: promote to "always check" (run automatically, don't just list)
  - Criteria never triggered after 10+ evaluations: suggest pruning
  - New failure pattern found: flag it and propose a new criterion. Don't add silently.

## System Review Schedule

Last system review: not yet

Periodically, suggest a system review. Check the date above — if 2+ weeks have passed or the project just hit a milestone, suggest it before starting the next task:
  - Prune stale rules in /knowledge/ that haven't been applied in 30+ days
  - Check if any hypothesis has enough evidence to promote or enough contradictions to discard
  - Review decision outcomes — did the trade-offs play out as expected?
  - Evaluate quality criteria: promote frequent triggers, flag never-triggered for pruning
  - Report what changed and why
  - Update "Last system review" date above

Don't run this automatically. Suggest it. The user decides when the time is right.