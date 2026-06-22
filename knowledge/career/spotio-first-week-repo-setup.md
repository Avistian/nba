# SPOTIO — First-Week ML/Data Repo Setup Playbook

The concrete day-one substrate to walk into SPOTIO with the AI-tooling multiplier already in place, so
one person (solo ML + data engineering) can do team-scale work. Goal: a repo that is **agent-legible**
and **eval-gated** before any modeling, mirroring the discipline already proven in this NBA prototype.

The ordering principle: **build the substrate that makes agents reliable and models trustworthy, then
model.** Don't open with relational GNNs — open with infra + a simple, well-measured baseline.

## Day 1–2 — Repo skeleton + agent legibility

1. **`AGENTS.md` at the repo root.** Encode: stack, conventions, how to run tests/lint, the eval
   standard ("a model change ships only if it beats baseline on the offline eval + clears a gate"),
   data-privacy rules (no PII/protected fields in features), and "automate-what's-verifiable / keep
   judgment" boundary. This is the single highest-ROI file — every agent run reads it.
2. **`.cursor/rules/` (or skills)** for repeatable tasks: "add a feature to the pipeline", "write a
   model card", "add an eval metric". Treat prompts/specs as reusable artifacts.
3. **One-command dev env**: `uv` (or `poetry`) + `Makefile` with `setup`, `check` (lint + types +
   tests), `eval`, `train`. Pin Python. Agents and future teammates both benefit.
4. **Strong typing + lint from commit 1**: `ruff`, `pyright`/`mypy` strict on `src/`, `tests/`,
   `scripts/`. Legible repos → dramatically better agent output.

## Day 2–3 — Data plumbing + schema introspection

5. **Profile the production data first** (answers career hypothesis H1). Stand up a read-only path /
   warehouse connection and an introspection script: tables, row counts, key relationships
   (rep ↔ lead ↔ household/account ↔ territory ↔ time), null rates, label availability, leakage risks.
6. **MCP database connector** so agents can safely introspect the real schema and draft pipeline code
   against it. This is a big multiplier on the data-engineering hat — the most automatable, most
   burnout-prone load.
7. **Reproducible feature/data pipeline** (e.g. `dbt` or a typed Python pipeline) with a small
   versioned sample for fast local iteration. Agents scaffold; you review the diff and the data
   contracts.
8. **Feature allow-list** (mirror this repo's ethics rail): no protected/geo/identity fields; review
   gate so a GNN/model can't sneak leakage in via message passing or joins.

## Day 3–5 — Eval harness BEFORE modeling (the trust + verification layer)

9. **Offline eval harness** keyed to **business metrics** (conversion lift, revenue per visit, rep-time
   saved), not just AUC. This is the triple-duty skill: it verifies agent-written code, builds trust at
   the job, and gives research rigor.
10. **A promotion gate** — a model ships only if it beats baseline on the primary metric *and* clears a
    statistical/uncertainty bar (the analog of this repo's DR lower-bound OPE gate). For
    next-best-action over logged data, plan for **off-policy evaluation** (propensity logging, overlap,
    IPS/DM/DR) so you can estimate a new policy's value honestly before any online test.
11. **Decision→outcome logging** from day one: log every recommendation + propensity + realized outcome
    so lift is provable later ("my model made $X"). Append-only.
12. **Regression test** that the default/baseline path stays stable, so agent-driven changes can't
    silently degrade it.

## End of week 1 — First model + first proof

13. **Ship the boring baseline**: gradient-boosted trees on clean tabular features for next-best-action,
    run through the eval harness, with one honest number vs a naive heuristic. This earns ownership
    credibility (R5) — do **not** start with RDL/tabular FMs.
14. **A one-page model card / writeup** (internal) — doubles as the seed for the first *external* public
    artifact (~month 6–9) once generalized and stripped of proprietary data.

## What to hand to agents vs. keep (R7)

| Delegate to agents | Keep for yourself (judgment) |
|--------------------|------------------------------|
| Pipeline + dbt/IaC scaffolding | What the model should optimize |
| Schema exploration via MCP | Eval design + metric choice |
| Feature-engineering boilerplate | Interpreting results, deciding ship/no-ship |
| Test + docs generation | Money-/privacy-/security-sensitive correctness |
| Migrations, backfills, refactors (background/cloud agents) | Choosing the modeling approach |

## Workflow habits to build (R6)

- **Plan mode → review the plan (cheap) before the diff (expensive).**
- **Execute against the eval/test harness you own** — never trust agent output you didn't verify.
- **Parallel / background / cloud agents** for independent grunt work (one scaffolds a pipeline,
  another writes tests, another drafts docs) using git worktrees so they don't collide. This is how one
  person becomes a team — and how grunt work runs while you're with your kid.
- **Critical diff review** as a trained muscle: read agent output adversarially ("where is this subtly
  wrong?").

## Definition of done for week 1

Not a fancy model — a **trusted substrate**: agent-legible repo, profiled data, a reproducible pipeline,
an eval harness + promotion gate, decision/outcome logging, and one simple baseline with an honest
number. Everything ambitious (RDL, tabular FMs) is then a safe, measurable experiment on top.
