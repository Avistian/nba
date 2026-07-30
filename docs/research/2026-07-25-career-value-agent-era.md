# Positioning Yourself When Coding Agents Do Most of the Work

Research synthesis for AI/ML engineers and adjacent roles. Date: 2026-07-25.

## Verdict

Agents are compressing the **floor** of engineering (implementation, boilerplate, standard
pipelines). Value is concentrating at the **ceiling**: defining what “done and safe” means,
binding models to proprietary objectives, and being the accountable operator who can prove work
belongs in production. The winning move is not competing with agents at typing — it is becoming
the person who **directs, verifies, and owns** systems agents cannot be held responsible for.

---

## 1. What the evidence actually says

### Labor market: bifurcation

- Stanford/others’ “Canaries in the Coal Mine” finds **entry-level employment declines** in
  occupations where generative AI use is predominantly **automative**, with muted effects where use
  is **augmentative**.
- Market snapshots through late 2025 / early 2026 show junior software postings falling sharply
  while senior roles and compensation are more resilient — consistent with “fewer people writing
  CRUD, more premium on people who can architect and supervise.”
- LeadDev’s 2025 AI Impact survey: engineers expect demand for **critical thinking** and
  **architectural design**; skills they prioritize learning are **managing agents** and prompt/
  agent interaction. Many also expect **less junior hiring** long-term.

### How AI is actually used in software

Anthropic’s Economic Index (software development report):

- Coding agents skew heavily **automative** (~79% on Claude Code vs ~49% on general chat).
- Even then, software work shows elevated **feedback-loop** patterns — humans still validate,
  return errors, and iterate. Automation of _implementation_ ≠ removal of _oversight_.

Implication: your job does not disappear when agents write code; it **moves** to the parts of
the loop that create trust.

### Expert framing that keeps recurring

| Source                                            | Frame                                                                                                                                                                                                                                                    |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Karpathy (Sequoia Ascent 2026)                    | Vibe coding raises the floor; **agentic engineering** raises the ceiling. Scarce: understanding, taste, eval design, security, boundaries, orchestration. Cheap: generation, API recall, boilerplate. “You can outsource thinking, never understanding.” |
| Pragmatic Engineer                                | Declining premium on polyglot/stack expertise; rising demand for **tech-lead traits** and product-minded engineering.                                                                                                                                    |
| Nicholas Zakas                                    | Coder → **orchestrator**: flight plans across agents, model selection, precise prompts, merge discipline.                                                                                                                                                |
| arXiv “Skills for the future software profession” | Bottleneck shifts from production to **verification & validation**; manage **cognitive debt** as AI authors more of the artifact surface.                                                                                                                |
| Floor/ceiling frameworks                          | AI commoditizes procedural work; tacit judgment remains human.                                                                                                                                                                                           |

---

## 2. The ML-specific barbell (avoid the middle)

For people who identify as AI/ML engineers, the sharpest map is:

```
[ Deep objective-bound ML ] ——— thin middle ——— [ AI-native rigor ]
   ranking, bidding,           "standard train     evals for stochastic
   calibration, OPE,           and ship pipeline"  systems, orchestration,
   online/offline gap          ← agents eat this   serving, failure triage
```

**Left weight (durable):** math over _your_ data and _your_ constraints. Frontier APIs do not know
your auction dynamics, label noise, or business tradeoffs. Retrieval/content understanding is
increasingly general-purpose; what stays yours is ranking under a goal, calibration, decision
logic, and online/offline honesty.

**Right weight (growing, starved for ML discipline):** people ship prompts that “look good.” Your
leakage reflex, adversarial holdouts, slice metrics, and “too-good-to-trust” flinch transfer
almost 1:1 to LLM-as-judge, agent traces, and prompt/tool sweeps.

**Middle (dangerous):** being the person who wires standard pipelines that an overnight agent can
scaffold. That relief is not safety.

---

## 3. What to become: six durable positions

Pick one primary identity; stack a secondary.

1. **Proof / evaluation owner** — Design machine-checkable acceptance: tests, adversarial splits,
   slice gates, promotion criteria, process-level agent eval (not only final-answer scores).
2. **Agentic systems engineer** — Tool contracts, plan/execute separation, observability of
   traces, cost/latency, permissions, recovery from loops and bad tool calls.
3. **Objective-bound applied ML specialist** — Own a business-critical predictor/policy loop end
   to end; use agents as lab techs for ablations, not as replacements for judgment.
4. **Product-minded technical lead** — Translate ambiguous stakeholder intent into specs,
   non-goals, and tradeoffs; decide what _not_ to automate.
5. **Domain-coupled hybrid** — Vertical depth (ads, clinical, logistics, security, finance) +
   agentic execution. Domain is a moat coding agents do not have.
6. **Accountability / release operator** — Sit between model, repo, security, and release lane;
   make “looks done” vs “is done” your brand.

---

## 4. Skills portfolio (what to practice weekly)

### Raise the ceiling

- Write **specs** before prompts: invariants, failure modes, non-goals, acceptance tests.
- Read **unfamiliar diffs at speed**; treat code review as the primary craft.
- Build **eval harnesses** for non-deterministic systems (pass-rates, worst-slice, adversarial).
- Practice **architecture under speed**: agents go fast; your boundaries must be reflexes.
- Run **agentic experiment loops**: agent sweeps → you decide what’s real vs leakage.
- Own **one production feedback loop**: metrics, incidents, online/offline gaps.

### Keep just enough floor fluency

- Use agents daily so you know their jagged failure modes (Karpathy’s “ghosts”).
- Do not outsource understanding of systems you are accountable for.

### Deprioritize as identity

- Being the fastest at handwritten syntax or the broadest language polyglot.
- Collecting model/API trivia without shipping proof artifacts.
- Pure “prompt engineer” branding without evals, tools, or domain.

---

## 5. 90-day positioning plan

**Days 1–30 — Reframe and prove the reflex**

- Ship a reproducible eval for a stochastic task (LLM/agent or classic ML) with an adversarial
  split and ground truth.
- Drive one real end-to-end agent task in a production-like repo; keep a log of what you refused to
  trust.

**Days 31–60 — Build the loop**

- Wire an agent-driven ablation/sweep that you judge (hypothesis → run → table → next experiments).
- Add one deterministic “scaffold gate” (no-LLM tests) around something agents touch.

**Days 61–90 — Ship opinionated proof**

- Narrow win: fine-tune, fusion model, or agent workflow that beats the naive API baseline **with
  numbers** (especially worst-slice).
- Publish a writeup with evals — the artifact is the positioning.

---

## 6. How to signal “worthy” to employers and collaborators

Portfolio that reads as ceiling, not floor:

| Weak signal                              | Strong signal                                               |
| ---------------------------------------- | ----------------------------------------------------------- |
| “I use Cursor/Claude daily”              | “Here is the eval harness and promotion gate I own”         |
| Demo video of agent coding               | Diff + tests + incident/postmortem notes                    |
| Fine-tuned model card with rosy averages | Adversarial / slice analysis that almost blocked a bad ship |
| List of frameworks                       | Owned production metric and online/offline diagnosis        |
| “Full-stack AI engineer”                 | Clear barbell: domain objective _or_ reliability systems    |

Narrative to practice: _I make agent output trustworthy under real constraints._

---

## 7. Adjacent paths if you are not “pure” ML

Still high demand:

- **Data / measurement** — labels, instrumentation, causal impact of agent rollouts
- **Security & governance for agents** — prompt injection, tool permissions, audit trails
- **MLOps / platform** evolved into **agent ops** — traces, cost, model routing, rollback
- **Technical product** — intent, prioritization, and acceptance criteria for agent-built features
- **Research eng / applied science** — where verifiability is partial and taste is load-bearing

---

## 8. Honest risks

- Entry paths that were “learn by writing lots of CRUD” are thinner; apprenticeship and
  judgment-building must be deliberate.
- More generated code without stronger practices increases **incident and cognitive-debt risk** —
  weak engineering habits hurt faster.
- “Learn AI tools” alone is not a moat; undifferentiated middle ML and middle software both get
  squeezed.

---

## Bottom line

Stay valuable by becoming scarce where agents are weak: **intent → specification → verification →
accountability**, especially on **proprietary objectives and production reality**. Use agents to
raise your iteration speed; never let them replace your understanding or your standards of proof.
