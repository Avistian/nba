# Agentic Workflow — Capability verification

Agent-driven E2E: the **agent exercises** the capability, a **fresh-context agent judges** evidence.

## Why not just `npm run test:e2e`?

Fixed scripts break down for **new capabilities** — paste-from-clipboard, camera input, novel ML
metrics, multi-service flows, "does this game map feel playable". The article's author scored 50+
layout experiments with a custom evaluator. The versatile pattern is:

1. **Exercise** — agent uses shell, browser MCP, APIs, custom scripts — whatever fits.
2. **Judge** — separate agent reads only acceptance criteria + artifacts (no authoring bias).
3. **Escalate** — `needs_human` when exercise is blocked or product behavior is ambiguous.

## Quick start

```bash
# One-off capability (any repo)
python3 scripts/agent_verify.py \
  --capability "User can export territory CSV" \
  --acceptance "CSV has header row and >=1 data row" \
  --exercise-hint "uv run python scripts/my_territory_demo.py --export /tmp/out.csv"

# Multi-capability spec (commit as .cursor/capability-verify.json)
python3 scripts/agent_verify.py --spec .cursor/capability-verify.json

# Wired into no-mistakes (auto-detects .cursor/capability-verify.json)
scripts/no_mistakes.sh -y
```

## Spec format (JSON)

Copy `.cursor/capability-verify.example.json` → `.cursor/capability-verify.json` and edit.

| Field | Purpose |
|-------|---------|
| `setup.shell` | Optional one-shot before exercise (start server, install deps) |
| `capabilities[].id` | Directory name under `output_dir` |
| `capabilities[].description` | What the capability is |
| `capabilities[].exercise.hints` | How the exercise agent should try it |
| `capabilities[].exercise.tools_allowed` | Reminder of available tools (browser MCP, shell, …) |
| `capabilities[].acceptance` | Judge checks these against artifacts |
| `capabilities[].min_score` | Judge score threshold (default 1.0 if acceptance listed) |
| `teardown.shell` | Optional cleanup |

## Artifact contract

Each capability writes under `artifacts/capability-verify/<id>/`:

- `exercise.log` — step-by-step what the agent did
- `exercise-summary.json` — `{"exercised": bool, "artifacts": [], "blockers": [], "notes": ""}`
- Any evidence files (screenshots, API JSON, stdout captures, custom evaluator output)

The runner emits `artifacts/capability-verify/manifest.json` with per-capability judge verdicts.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All capabilities passed |
| 1 | Failed (exercise or judge) |
| 2 | `needs_human` — blocked exercise or ambiguous product behavior |

`no_mistakes.sh` treats exit 2 as escalation (same as ambiguous code review).

## Trust guardrails (same as code review)

- Exercise and judge are **separate** `agent -p` invocations.
- Judge never sees the exercise session transcript — only summary + artifacts + diff.
- `needs_human` never auto-passes.
- You still merge manually; the manifest is audit evidence on the PR.

## Browser / MCP capabilities

For UI features, add exercise hints like:

```json
"hints": [
  "Start dev server: npm run dev",
  "Use browser MCP (chrome-devtools) to open http://localhost:3000",
  "Screenshot each step to artifacts/capability-verify/<id>/",
  "Capture console errors via MCP"
]
```

See `.cursor/skills/browser-testing-with-devtools/SKILL.md`.

## Custom evaluators (scoring loops)

For measurable metrics (latency, map quality, model accuracy), the exercise agent can run **your**
script and save JSON; the judge checks acceptance against that file:

```json
"hints": [
  "Run: python evaluators/score_layout.py --out artifacts/capability-verify/map/score.json"
],
"acceptance": [
  "score.json exists with score >= 0.8",
  "No regression vs baseline in score.json"
]
```

This is the article's "50+ experiments scored by simulation" pattern — the agent orchestrates,
your evaluator scores.

## Portable / any repo

- `scripts/agent_verify.py` uses **stdlib only** (no uv required to run the verifier itself).
- Point `AGENT_BIN` at the Cursor CLI `agent` binary.
- Set `AGENT_VERIFY_MOCK=pass|fail|needs_human` for CI control-flow tests without API calls.
