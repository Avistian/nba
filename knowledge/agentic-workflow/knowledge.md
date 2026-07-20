# Agentic Workflow — Knowledge

Source article: *An Ex-Meta L8's Agentic Engineering Setup* — Kun Chen, ByteByteGo, Jun 2026.
The author works terminal-first (WezTerm + tmux + Neovim + Claude Code/OpenCode) and built
custom CLI tools to remove friction. This doc translates his two headline tools into a
**Cursor-IDE-native** workflow, since we are IDE users, not terminal purists.

## The two tools, in one paragraph each

- **`gnhf` ("good night, have fun")** — a long-running orchestrator for tasks too big for one
  context window. It splits the task into small steps; each step runs in a *fresh* context
  seeded with a shared base context + learnings from prior steps. Failed attempts roll back
  automatically and inform the next attempt. A token budget bounds cost. Output: a branch of
  well-organized commits + a `notes.md` summary. Same family as the "Ralph loop" / autoresearch
  patterns.

- **`no-mistakes`** — an autonomous *validation* pipeline you run after an agent finishes.
  `no-mistakes -y` commits with a conventional message onto a descriptive branch, rebases onto
  latest `main` and resolves conflicts, spins up **fresh-context** agents to peer-review and
  self-correct obvious bugs, tests the change **end-to-end** and produces evidence, closes doc
  gaps, fixes linting, pushes, opens a structured PR, and babysits CI to green. Everything runs
  autonomously *except decisions it deliberately escalates to the human*. The author reports 68%
  of changes it processed had bugs it caught.

The article's supporting cast (for context): `treehouse` (worktree pool for parallelism),
`Lavish Editor` (interactive HTML plans), Tailscale + mosh + tmux (remote control).

## Why this matters for the three asks

| Ask | Article mechanism | Cursor-native equivalent |
|-----|-------------------|--------------------------|
| Automated validation | `no-mistakes` pipeline | Cursor Hooks + headless `agent -p` reviewer + Bugbot + `make check`/E2E |
| Parallelization | tmux windows + `treehouse` worktrees | Background/Cloud Agents (one per PR/branch) + git worktrees |
| Without sacrificing trust | fresh-context review, human escalation, E2E evidence, audit log | same three, enforced by hooks + a strict escalation contract |

## The trust model (this is the whole point)

The author's leverage does **not** come from letting agents merge freely. It comes from four
guardrails. Adopt these verbatim; they are the load-bearing ideas:

1. **Fresh-context review.** The agent that wrote the code is biased toward believing it works.
   Always review in a *new* session/agent that only sees the diff, not the authoring transcript.
2. **Escalate ambiguous, product-changing decisions to the human.** Auto-fixing every "finding"
   lets a reviewer agent drift into rabbit holes. Anything that changes product behavior or is
   ambiguous stops and asks you.
3. **Force end-to-end evidence.** Passing unit tests is necessary but not sufficient — models
   over-trust them. Require a run of the real thing (demo/app/CLI) plus an artifact (log,
   screenshot, output) attached to the PR.
4. **Everything is auditable.** Every auto-fix is a separate commit with a conventional message;
   every escalation and every piece of evidence is logged on the PR so you can reconstruct what
   happened at a glance.

A useful mental model from the article: **you are the engineering manager; the agents are the
team.** Managers rarely read every line — they require peer review + evidence before shipping.

## Cursor primitives you build on (verified Jul 2026)

- **Headless CLI:** `agent -p "<prompt>"` runs non-interactively; add `--force`/`--yolo` to let
  it edit files, `--output-format json` for parseable output, `--model <slug>`. Set
  `CURSOR_API_KEY` for unattended runs. The binary is `agent` (install:
  `curl https://cursor.com/install -fsS | bash`). There is **no** `cursor lint/review/...`
  subcommand — everything is a prompt to `agent -p`. Prepend `&` to push a conversation to a
  **Cloud Agent** that keeps running while you're away (pick up at `cursor.com/agents`).
- **Cursor Hooks** (`.cursor/hooks.json`): lifecycle events including `afterFileEdit`, `stop`,
  `beforeShellExecution`, `postToolUse`, `subagentStop`, `beforeSubmitPrompt`. Command hooks can
  **block** with exit code `2` on supported events (`beforeShellExecution`, `beforeReadFile`,
  `preToolUse`, ...). `afterFileEdit`/`stop`/`beforeSubmitPrompt` are effectively observational
  for the agent's decision-making but are perfect for running formatters/linters/gates.
- **Background & Cloud Agents:** run autonomously on their own branch and open a PR — the native
  replacement for "a tmux window per task."
- **Bugbot:** automated PR reviewer — a ready-made "fresh-context reviewer" that already lives
  where PRs are.
- **Subagents (Task tool), slash commands (`.cursor/commands/`), rules (`.cursor/rules/`),
  `AGENTS.md`:** the in-editor glue.
- **Git worktrees:** the `treehouse` idea; Cursor works fine across worktrees.

## `gnhf` → Cursor design

Two ways to get it, cheapest first:

**A. Just use a Cloud/Background Agent.** For most "big but boundable" tasks, hand the plan to a
Cloud Agent (`& agent "implement PLAN.md ..."` or the dashboard). It already runs long, commits
to a branch, and opens a PR. Use this before building anything.

**B. `scripts/gnhf.sh` (shipped)** — for tasks that overflow a single context or need
rollback-on-failure + a token budget, which a single agent session does not give you. Design:

```
gnhf.sh "<goal>"  [--budget-usd N] [--max-steps N] [--validate "make check"]
                  [--base-context FILE] [--branch-prefix gnhf/] [--model SLUG]
```

Loop, one step per fresh `agent -p` invocation:
1. Seed the step with: base context (repo conventions / `AGENTS.md` / the plan) + the running
   `notes.md` (learnings so far) + "do the **next smallest** step toward GOAL; print `GNHF_DONE`
   when the goal is fully met."
2. Run the validation gate (default `make check`).
3. **Pass →** `git commit` the step (conventional message). **Fail →** `git reset --hard`
   (rollback) and append the failure + captured error to `notes.md` so the next fresh step avoids
   it.
4. Accrue estimated cost from JSON output; stop at `--budget-usd`, `--max-steps`, or `GNHF_DONE`.
5. Emit `notes.md` summary. You wake up to a branch of clean commits.

This is deliberately model-agnostic (swap `--model`), matching the article's "avoid vendor
lock-in" stance.

## `no-mistakes` → Cursor design

**`scripts/no_mistakes.sh` (shipped)**, invoked by hand, by the `/no-mistakes` slash command, or
by `gnhf` at the end of a run:

```
no_mistakes.sh [-y] [--base main] [--e2e "make demo"] [--no-push]
```

Pipeline (each agent step is a fresh `agent -p`, so review is unbiased):
1. **Glance gate (human, 10s).** Print the diffstat first — the article's "scan the diff, make
   sure it's roughly what I asked." Without `-y`, wait for confirm.
2. **Commit + branch.** Ask the agent to author a Conventional Commit message from the diff;
   create a descriptively named branch.
3. **Rebase onto latest base.** Fetch + rebase; if conflicts, an agent step resolves them, then
   re-validate.
4. **Fresh-context peer review.** `agent -p` sees only the diff and returns structured JSON:
   `{ obvious_bugs: [...], ambiguous: [...] }`. Obvious bugs are auto-fixed as **separate
   commits**; anything `ambiguous`/product-changing is printed and the pipeline **stops and
   escalates** (trust guardrail #2).
5. **E2E evidence.** Run the configured E2E command (`make demo`, the API, a scenario script),
   capture stdout/artifacts into `artifacts/no-mistakes/`, and require it to exist before
   proceeding (guardrail #3).
6. **Docs + lint.** `make fmt` + a doc-gap agent pass.
7. **Push + PR.** Push; open a PR whose body includes the auto-fix log and a **Testing** section
   linking the evidence (guardrail #4). Locally the user can use `gh pr create`; in this repo,
   PRs are opened via the ManagePullRequest tool.
8. **Babysit CI.** Poll CI; on red, a fresh agent step proposes a fix, re-run from step 5.

### Where Cursor gives it to you for less work
- **Bugbot** already does step 4 on the PR — you can lean on it and keep `no_mistakes.sh`
  focused on commit/rebase/E2E/CI so you are not reinventing review.
- **Hooks** move the cheap parts *left* so `no-mistakes` rarely finds lint/format issues:
  - `afterFileEdit` → `ruff format` the edited file (see `scripts/hooks/format_edited.sh`).
  - `stop` → run a fast lint/type gate and remind the agent to produce E2E evidence
    (`scripts/hooks/stop_gate.sh`); make it blocking with exit code `2` once you trust it.

## Parallelization without losing the thread

- **One task = one Background/Cloud Agent = one branch = one PR.** This is the IDE-native version
  of "a tmux window per task." Status is visible in the agents list; you review PRs, not panes.
- **Worktrees** (`treehouse` idea) only matter for *local* parallel work sharing one checkout;
  Cloud Agents sidestep it entirely by running in isolated environments.
- Keep tasks independent (see the `dispatching-parallel-agents` skill) so 5–10 can run at once
  without stepping on each other. `gnhf` per task means each returns a clean PR you audit later.

## Concrete adoption path for this repo (NBA project)

1. Turn `.cursor/hooks.example.json` into `.cursor/hooks.json` → format-on-edit + stop-gate. Low
   risk, immediate.
2. Use `/no-mistakes` (or `scripts/no_mistakes.sh`) after any agent change; `--e2e "make demo"`
   is the natural E2E surface here, plus `make check` for the lint/type/test gate that
   `quality/criteria.md` already blocks on.
3. For phase-sized work (the repo grows in "phases"), use `scripts/gnhf.sh` overnight with
   `--validate "make check"` and the phase plan as `--base-context`.
4. Run independent phases/experiments as separate Cloud Agents → separate PRs; the
   experiment-leaderboard style scoring in `dataset-eval` is exactly the "evaluator scores each
   attempt" case the article calls out for `gnhf` batch experiments.

## What NOT to copy

- Don't auto-merge. Every guardrail above raises *what arrives at review*; the merge stays a
  deliberate human step.
- Don't let a reviewer agent auto-fix ambiguous/product decisions — that's how you lose trust.
- Don't rely on unit tests alone as "validation."
