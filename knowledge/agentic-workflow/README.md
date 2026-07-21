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
| [capability-verification.md](capability-verification.md) | Agent exercises + judges capabilities (versatile E2E) |
| [rules.md](rules.md) | Confirmed practices to apply by default |
| [hypotheses.md](hypotheses.md) | Ideas that still need real usage data |

## Portable toolkit: `flow/`

All tools live in **[`flow/`](../../flow/)** — copy that folder into any repo. See
[`flow/README.md`](../../flow/README.md) for setup and the full flow definition.

Quick start in any repo:

```bash
./flow/install.sh repo    # creates .cursor/flow.json + templates
export PATH="$(pwd)/flow/bin:$PATH"
```

## Starter kit (in `flow/`)

- `flow/bin/gnhf` — long-running fresh-context orchestrator
- `flow/bin/no-mistakes` — validate-review-evidence-PR pipeline
- `flow/bin/agent-verify.py` — agent exercises + judges capabilities
- `flow/bin/overnight` — parallel gnhf + no-mistakes fan-out
- `flow/hooks/` + `flow/templates/` — Cursor hooks and per-repo config
- `scripts/*.sh` — thin wrappers delegating to `flow/bin/` (this repo)
