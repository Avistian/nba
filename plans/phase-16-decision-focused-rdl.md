# Phase 16 — Decision-focused RDL: the one genuinely novel combination (DEFERRED / optional)

**Status: deferred — the research frontier.** This is where mixing RDL stops being "a maybe-better
predictor" and becomes a real contribution (doc 12 §7): train a **relational** value model
**end-to-end through an orienteering optimizer**, so the GNN learns to predict prizes that make the
*route* maximally valuable under a real time/road/team budget. Each half exists individually
([Phase 12](phase-12-decision-focused-learning.md) decision-focused learning + [Phase 14](phase-14-relational-deep-learning.md)
RDL); their composition on relational enterprise data, gated by honest OPE, is the fresh part.

**Depends on:** Phase 12 (decision-focused / SPO+), Phase 14 (RDL value model), Phase 9 (relational
data), and benefits from Phase 10 (the real OP objective). Grounded in
[docs/12 §7-8](../docs/12-relational-deep-learning-mixin.md) and [docs/20](../docs/20-decision-focused-rdl.md).

## Feature flags (when built)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `use_decision_focused_rdl` | `bool = False` | Train the GNN end-to-end through the optimizer. Requires `reward_model_kind="rdl"`. |
| `dfrdl_objective` | `Literal["spo","reinforce"] = "spo"` | SPO+ subgradient, or policy-gradient through the route. |
| (inherits) | — | All Phase 12 SPO+ knobs and Phase 14 GNN knobs apply. |

## What it composes (doc 12 §8 — the legitimate "spatio-relational" kernel)

- **Relational signal** -> the Phase 14 GNN encoder produces `q(x, a)` from the CRM/neighbor graph.
- **Spatial signal** -> real road travel times (Phase 10 OSRM) feed the optimizer; coordinate features
  enter the encoder via standard spatial embeddings (sinusoidal/Fourier, Space2Vec-style).
- **The fusion** -> happens in the **objective**: orienteering maximizes relational prize subject to
  the spatial budget, trained end-to-end (Phase 12's SPO+ loop with the GNN as the prize predictor).

## Sketch

```
src/nba/reward/decision_focused.py   # extend spo_finetune to accept a GraphRewardModel predictor
src/nba/reward/graph_model.py        # expose a differentiable prize head for the SPO+/REINFORCE loop
tests/test_decision_focused_rdl.py
```

## The rails still bind (doc 12 §7, §9)

- No "zero-hallucination" / "guaranteed maximum" claims — learned policies give neither.
- Calibration re-applied after end-to-end training; ethics allow-list at the graph layer; no oracle at
  serve time; **promotion only through the same DR gate on route value.**

## Leaderboard entry (lift/regression)

Records into the [Phase 17 leaderboard](phase-17-experiment-leaderboard.md): `phase16-dfrdl`
(`NBA_USE_DECISION_FOCUSED_RDL=1`, `NBA_REWARD_MODEL_KIND=rdl --dataset relational`). The bar is the
highest in the roadmap: it must beat **both** `baseline` **and** the better of `phase12-spo` /
`phase14-rdl` on the **primary metric** through the DR gate. Expected verdict **lift** only if the
fusion genuinely exceeds its two halves; otherwise **neutral** and you keep the simpler component.

## Acceptance (when built)

- The fused model beats both the LightGBM baseline and the standalone RDL model on **route value**
  through the DR gate on relational data; otherwise it is not promoted.
- All flags off / `reward_model_kind="lightgbm"` reproduces today; `ruff` / `pyright` clean;
  `pytest` green (skipped without the optional extra).
