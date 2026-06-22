# Career & Positioning — Knowledge

Facts and patterns about how the repo owner should position themselves as an AI/ML engineer in an
era of AI-assisted programming and possible AGI. Captured from the 2026-06-22 strategy conversation.

## Profile (as of 2026-06-22)

- Age 34, based in Gdańsk, Poland; staying there, working **remotely**.
- New role: **SPOTIO** (US startup, door-to-door field-sales SaaS). Owns **all ML features**,
  starting with **next-best-action**, then more. Initially **solo** (also doing the data engineering);
  may **lead a team** later once ML proves its ROI.
- Comp: **salary only** (B2B), ~30k PLN/month + VAT. No equity.
- Household: wife earns ~8k PLN gross (CoE); spend ~15k/month; saves ~5–8k/month.
- Savings: **0.5M PLN**. Goal: **Barista-FIRE at 45+** — cover bills without worry, then follow
  research interests / whatever is valuable.
- New parent (3-month-old). **Cannot go "all-in"**; wants to be present as a parent. This is a
  **hard constraint**, not a preference.
- Genuine research interest: **relational deep learning (RDL)** and **tabular foundation models**
  (e.g. RelBench/PyG-relational, TabPFN-style in-context tabular models).

## Priority ranking (owner-stated)

1. Recognized researcher (but "far away")
2. Elite IC
3. Financial resilience
4. Founder (lowest — explicitly not the goal)

## Key structural insight: the job *is* the research

Door-to-door field sales produces exactly the data structure RDL + tabular FMs are built for
(reps ↔ leads ↔ households ↔ territories ↔ time; sequences of touches → outcomes). Next-best-action
**is** a relational + sequential prediction problem over a tabular/graph schema. So the research
interest is plausibly the *best architecture* for the job — convergence to exploit, not a side hobby.
This repo (NBA prototype: bandits, OPE, reward model, routing, relational dataset, ethics, eval) is
already the applied arm of that interest.

## Financial reality (honest math)

Using owner numbers (0.5M PLN now, ~6.5k/month saved avg, 11 years to 45, ~5% real return):

- 0.5M → ~0.85M real; contributions add ~1.1M real; **total ≈ ~2.0M PLN at 45**.
- At 4% withdrawal ≈ **~6.5k/month** passive income.
- Against 15k/month spend: portfolio ~6.5k + wife ~5.8k ≈ 12.3k, ~2.7k gap covered by light work.

**Conclusion: Barista-FIRE at 45 is realistic on current trajectory; full FIRE by 45 is not.**
Financial resilience (#3) is therefore the **engine that funds the research dream (#1)**, not a rival
to it. The savings *rate* is the single biggest lever; there appears to be an unexplained gap between
net income and (spend + stated savings) worth understanding.

## Realistic definition of "recognized researcher"

Becoming a *frontier* researcher (inventing the next tabular FM) is unrealistic given the constraints
(newborn, solo, FIRE track, not all-in) and would damage the parenting/sustainability goals. The
achievable, arguably-better version:

> **Recognized applied-research voice at the bridge between RDL / tabular FMs and real-world
> relational business (field-sales) data.**

Open niche; owner has an asset few academics have — a production system with real, messy, large-scale
relational data and a live testbed. Reachable on a part-time cadence.

## AI-tooling thesis (Cursor / agents)

AI-assisted engineering lets one person do team-scale work — exactly what a solo ML+DE needs. It
(a) lets him ship ML *and* data-eng features alone, (b) buys back hours for parenting + research,
(c) accelerates the "lead a team" inflection. Code *generation* is commoditizing; the durable skills
are the **two ends** AI is bad at — **specification/intent** and **verification** — plus
**orchestration** of agents. Win as a **director + verifier** of AI, not a co-typist.

## Connection to the AGI scenario

In an exponential-capability world, no individual career strategy protects you as *labor* if general
competence is fully automated. The robust hedges that overlap with the plan above: **financial
resilience + ownership of assets** (here: FIRE portfolio), **judgment + real deployment experience +
named trust** (the niche reputation), and **meaning not tied to economic productivity** (parenting,
relationships — which the owner already prioritizes). The plan is deliberately robust across "AI
plateaus" and "AI keeps accelerating" because those hedges pay off in both.
