# Agentic Workflow — Domain

How to run an agent-directed development workflow (automated validation + parallelization
without sacrificing trust) using **Cursor** as the primary surface instead of a terminal-only
setup.

Seeded from the ByteByteGo post *"An Ex-Meta L8's Agentic Engineering Setup"* (Kun Chen,
Jun 2026), which describes three home-grown CLI tools — `gnhf`, `no-mistakes`, `treehouse` —
plus `Lavish Editor`. This domain re-derives those ideas as Cursor-native building blocks.

| File | Purpose |
|------|---------|
| [knowledge.md](knowledge.md) | The tool-by-tool mapping, designs, and trust model |
| [overnight-runbook.md](overnight-runbook.md) | Overnight gnhf + parallel agents playbook |
| [rules.md](rules.md) | Confirmed practices to apply by default |
| [hypotheses.md](hypotheses.md) | Ideas that still need real usage data |

## Starter kit shipped with this domain

- `scripts/gnhf.sh` — long-running fresh-context orchestrator (the `gnhf` idea)
- `scripts/no_mistakes.sh` — autonomous validate-review-evidence-PR pipeline (the `no-mistakes` idea)
- `scripts/agent_verify.py` — agent exercises + judges capabilities (versatile E2E)
- `scripts/overnight.sh` — parallel gnhf + no-mistakes fan-out (worktrees)
- `scripts/hooks/` — `afterFileEdit` formatter + `stop` gate for Cursor Hooks
- `.cursor/hooks.example.json` — copy to `.cursor/hooks.json` to activate
- `.cursor/commands/no-mistakes.toml` — `/no-mistakes` slash command
