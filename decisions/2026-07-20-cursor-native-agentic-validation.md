## Decision: Recreate `gnhf` + `no-mistakes` as Cursor-native tooling rather than adopting the terminal-first stack

## Context

The ByteByteGo post *An Ex-Meta L8's Agentic Engineering Setup* (Kun Chen, Jun 2026) describes a
high-throughput agentic workflow built on a terminal-first stack (WezTerm + tmux + Neovim +
Claude Code/OpenCode) with custom CLI tools: `gnhf` (long-running fresh-context orchestrator),
`no-mistakes` (autonomous validate → review → E2E-evidence → PR pipeline), `treehouse` (worktree
pool), and `Lavish Editor` (interactive HTML plans). We are a Cursor IDE user wanting automated
validation + parallelization without sacrificing trust, so we need the *ideas* re-hosted on
Cursor primitives, not the author's exact terminal tooling.

## Alternatives considered

1. **Adopt the author's stack wholesale** (tmux/Neovim/Claude Code + his CLIs). Highest fidelity,
   but abandons the IDE we already use and its integrations (Bugbot, Cloud Agents, hooks).
2. **Cursor-native re-implementation** — map each tool to Cursor primitives: headless
   `agent -p` for orchestration/review, Cursor Hooks for gates, Background/Cloud Agents +
   worktrees for parallelism, Bugbot for review, `make check`/`make demo` for validation/E2E.
   Ship thin shell scripts (`gnhf.sh`, `no_mistakes.sh`) + a slash command + a hooks example.
3. **Do nothing custom; rely only on Cloud Agents + Bugbot.** Lowest effort; covers parallelism
   and review but not fresh-context orchestration with rollback/budget, nor a forced E2E-evidence
   gate.

## Reasoning

Option 2 keeps us in the IDE we already use while capturing the article's load-bearing ideas
(fresh-context review, human escalation, forced E2E evidence, auditability). Cursor already
supplies most primitives (`agent -p`, hooks, Cloud Agents, Bugbot), so the custom surface is
small and model-agnostic. Option 1 throws away working integrations; option 3 misses the
overnight-orchestrator and evidence-gate that produce most of the quality win.

## Trade-offs accepted

- We maintain two shell scripts and a hooks config instead of using a polished third-party tool.
- The scripts' agent-invocation steps can only be fully exercised with a real `CURSOR_API_KEY`;
  control-flow (branch/rollback/budget/escalation) is tested here with a mock `agent`.
- Some overlap with Bugbot (peer review) is intentional until H2 (see hypotheses) is settled.

## Supersedes

(none — new area; routes from knowledge/INDEX.md → agentic-workflow/)
