# Career Plan — Short → Long Term (conversation summary, 2026-06-22)

Summary of the strategy conversation. Objective function: **freedom + meaning + "enough"** —
Barista-FIRE at 45 funding the freedom to do valuable/research work, while being a present parent.
See [knowledge.md](knowledge.md) for facts, [rules.md](rules.md) for principles,
[hypotheses.md](hypotheses.md) for open questions, and
[spotio-first-week-repo-setup.md](spotio-first-week-repo-setup.md) for the day-one playbook.

## The thesis in one paragraph

Drop founder/equity pressure — it doesn't serve the goals. The US-salary/Poland-cost setup makes
**Barista-FIRE at 45 realistic (~2.0M PLN → ~6.5k/month passive)**, so financial resilience is the
*engine* that funds the research dream, not a rival to it. Redefine "recognized researcher" from the
unrealistic frontier version to the very achievable **niche bridge-builder between RDL/tabular-FM
research and real production field-sales data** — a space where the job *is* the research material and
almost no academic has the data. Build that reputation as a sustainable ~1-artifact-per-quarter
side-stream that never threatens parenting time; sequence the job **trust → infra → ambitious models**;
master AI tooling to be a **director + verifier** so one person does team-scale work; negotiate
**publishing rights + salary** (not equity). Arrive at 45 holding financial freedom, a recognized niche
name, and lead-level credibility — none bought at the expense of the others or the kids.

## Phase 0 — Before / first weeks at SPOTIO

- Negotiate **publishing/open-source latitude** (methods + "lessons from production", no customer data)
  and **salary**. Skip equity. Get scope-of-ML-ownership clarity (team to follow once ROI shown).
- **Profile the data** (tests H1): volume, cleanliness, how relational/temporal it really is. This
  decides what's possible.
- Stand up the agent-legible repo substrate (see the first-week playbook): `AGENTS.md`, eval/test
  harness, one-command dev env.

## Short term (0–9 months) — trust, infra, don't over-engineer

- Ship a **simple, rigorously-evaluated next-best-action baseline** (gradient-boosted trees + clean
  features + honest offline eval, ideally a small online A/B). Earn trust fast.
- As solo ML+DE, the first real product is the **data pipeline + eval harness**, not a fancy model.
- Establish the **decision→outcome feedback loop** so lift is provable ("my model made $X").
- **First public artifact (~month 6–9):** low-risk, generalizable (e.g. "evaluating next-best-action
  for field sales", or an open-source relational-feature/eval tool). Establishes the voice without
  proprietary data.
- Use agents to absorb the data-engineering grunt (tests H5) so judgment time goes to modeling + the
  writeup, and parenting time is protected.

## Medium term (9–30 months) — bring the research in, build the name

- Introduce **relational deep learning** where it beats the baseline, measured honestly (rep ↔ lead ↔
  territory ↔ time graph). Engage the **RelBench / PyG-relational** ecosystem; contributing to their
  OSS/benchmarks is the cheapest fast route to recognition without a PhD.
- Use **tabular foundation models (TabPFN-style)** for cold-start regimes (new territory/product/rep/
  customer) — a real business pain *and* a publishable applied-research angle.
- Continue ~quarterly public artifacts; target within this window: a couple of solid writeups, one OSS
  tool people use, and ideally **one workshop paper or conference talk** on RDL/tabular-FM applied to
  real relational business data → "recognized" in the niche sense.
- At the ROI proof point, **start leading a team** (buys back time, raises IC→lead profile,
  protects parenting). Renegotiate scope with maximum leverage right after a demonstrated win.

## Long term (30 months → 45) — glide into Barista-FIRE with optionality intact

- Hold simultaneously: **financial freedom (#3)**, a **recognized niche reputation (#1, achievable
  form)**, and **elite-IC/lead credibility (#2)** — all from the same path, no forced choice.
- At Barista-FIRE: choose part-time consulting/research, an industrial PhD (if the frontier still
  calls), OSS/research contributions, advising — whatever is valuable. Founder (#4) stays available,
  unforced.

## Honest risks to watch

- **Identity mismatch:** if the owner truly wants *frontier* research, current constraints don't
  support it; that would require changing the life (funded PhD / full-time lab), conflicting with
  parenting + FIRE. Confirm internally that the niche version satisfies.
- **Over-engineering year one** — the biggest risk; earn trust with simple models first.
- **Solo-ML burnout** — automate DE, guard scope, push for a team at ROI.
- **Lifestyle creep** — erodes the FIRE engine; lock in a high savings rate.
- **Publishing rights unknown** — clarify early; gates the #1 goal.

## Dependencies / open questions (see hypotheses.md)

Data richness (H1), publishing freedom (H2), savings-rate gap, salary range. Sharing the salary +
invested-savings numbers would let the FIRE timeline and possible earlier hour down-shift be made exact.
