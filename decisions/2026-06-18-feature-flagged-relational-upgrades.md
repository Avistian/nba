# Decision: Ship the docs 11/12 improvement roadmap as feature-flagged phases, relational dataset first

## Decision

Plan the NBA improvement roadmap (the upgrades in
[docs/11](../docs/11-improving-nba-spatio-relational-optimization.md) and
[docs/12](../docs/12-relational-deep-learning-mixin.md)) as eight new phases (9–16). Two binding
constraints:

1. **Every upgrade is a feature flag, off by default.** Each phase adds `NBA_*` settings whose
   defaults reproduce today's verified behavior exactly. Heavy dependencies (PyTorch/PyG, the neural
   router) are optional extras, never imported on the default path.
2. **The dataset becomes relational first, as a *new* dataset that mirrors the flat one.** Phase 9
   adds `dataset_mode="relational"` emitting a schema-identical `BanditEvent` stream alongside the
   existing flat simulator, rather than rewriting it.
3. **Every flag must prove itself on an append-only experiment leaderboard.** Phase 17 records each
   experiment's metrics, delta vs the `baseline` (all flags off), DR-gate result, and a
   **lift/regression/neutral** verdict. A *lift* requires both a higher primary metric (realized
   shift value) and clearing the DR gate, so an upgrade is adopted only on logged evidence.

## Context

Docs 11 and 12 propose value-side (relational deep learning, decision-focused learning) and
optimizer-side (orienteering, risk-aware, dynamic, neural CO) upgrades to a system whose 0–8 loop is
fully implemented and verified (143 passing tests, OPE-gated promotion, ethics rails). We needed a
roadmap that adds these without destabilizing the proven loop, and doc 12 §5.2 is explicit that RDL
is pointless until the data carries real relational structure.

## Alternatives considered

- **Rewrite the simulator to be relational in place.** Rejected: it would churn the substrate of the
  whole test suite and remove the ability to benchmark RDL against the flat-data LightGBM baseline.
- **Adopt upgrades directly (no flags).** Rejected: violates the repo's reversibility/OPE-gate
  discipline and makes regressions hard to isolate.
- **Lead with the high-value research (decision-focused / RDL) before the cheap wins.** Rejected:
  doc 11 §9 and doc 12 §6 both argue the cheap, sure optimizer wins come first.

## Reasoning

Flags + a mirrored dataset make each upgrade a falsifiable, reversible experiment with a clean
off-ramp: nothing changes until opted in, and every new value model is just another candidate that
must clear the same DR promotion gate on route value. Front-loading the relational dataset unblocks
RDL while keeping a fair head-to-head against LightGBM.

## Trade-offs accepted

- Some duplication between the flat and relational simulators (mitigated by reusing context/ames/
  distance helpers).
- Maintaining two dataset paths and more config surface area, in exchange for safety and honest
  evaluation.

## Supersedes

None (extends PLAN.md "Resolved decisions" 1–4 with items 5–6).
