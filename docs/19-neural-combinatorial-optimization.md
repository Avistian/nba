# 19 — Neural combinatorial optimization (deferred / optional)

> The companion doc for [Phase 15](../plans/phase-15-neural-combinatorial-optimization.md). **This is
> deferred** — specified so the seam is understood, not to be built speculatively. Read
> [11-improving-nba-spatio-relational-optimization.md](11-improving-nba-spatio-relational-optimization.md)
> §7.

This is **Upgrade 4**: high cost, adopt only with a measured reason.

## 1. What it is, from zero

**Neural combinatorial optimization (NCO)** trains a neural network to *emit* solutions to
combinatorial problems directly, instead of searching with a classical solver. For routing the
canonical recipe is an **attention-based encoder-decoder** that reads the nodes (coordinates + prizes)
and **autoregressively points at the next node to visit**, trained by reinforcement learning to
maximize route value (Kool et al. 2019; POMO, Kwon et al. 2020). Notably the standard attention model
already solves TSP, CVRP, the Orienteering Problem, and the Prize-Collecting TSP.

## 2. Why you probably don't need it yet

OR-Tools solves neighborhood-sized instances (tens to low hundreds of doors) to near-optimality in
seconds, **deterministically**, with first-class time windows and capacities. A learned policy trades
determinism and optimality *guarantees* for **speed at scale** and amortized solving (doc 11 §7.2).
Build it only when:

- you must (re)route **thousands** of doors **many times per minute**, or
- you want the router to exploit statistical regularities across your specific neighborhoods, or
- you're pursuing the **end-to-end** vision where value model + router are one network
  ([20-decision-focused-rdl.md](20-decision-focused-rdl.md)).

If none hold, keep OR-Tools.

## 3. The honest caveat (bake it into every claim)

A learned router produces **heuristic** solutions: usually excellent, **never provably optimal**. Never
advertise "the mathematically guaranteed best route" from an RL policy (doc 11 §7.3). If you build it,
keep OR-Tools as the **reference oracle in tests** and assert the learned router stays within an
acceptable optimality gap on held-out instances — that is how you stay honest.

## 4. The seam (when built)

A `Router` protocol (`solve(coords, profits, time_matrix, ...) -> Route`) wraps both the existing
OR-Tools solver and a new `neural_router.py`. The flag `router_kind` selects between them, defaulting
to `"ortools"` (today). The neural encoder can optionally consume the Phase 14 GNN embeddings as its
front end — the bridge to the fusion in [Phase 16](../plans/phase-16-decision-focused-rdl.md).

> See also: [20-decision-focused-rdl.md](20-decision-focused-rdl.md).
