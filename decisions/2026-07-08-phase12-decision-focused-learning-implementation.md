## Decision: Phase 12 trains the reward model on route value via two on-ramps — a gradient-free boundary reweighting and an SPO+ **linear correction head** — behind one `QModel`-preserving flag; the head is standardized-space-fit (so it can't explode), SPO+'s "true prize" is the realized logged reward (oracle-free), and decision-focused learning applies to the served point model only, not the bootstrap ensemble

## Context

Phase 12 (Upgrade 2) stops training the reward model to win a squared-error contest and starts
training it to make good include/skip/order decisions. Only doors near the router's include/skip
boundary can change the route, so that is where model capacity should go. The change had to stay
behind a flag that reproduces today's model exactly, keep the served `QModel` interface (so the
orchestrator, API, bandits, and OPE do not move), and never touch a simulator oracle (the new module
lands in `src/nba/reward/`, scanned by the no-oracle-leak AST guard). Several forks had to be
resolved, and the first SPO+ implementation was numerically broken.

## Alternatives considered

- **SPO+ predictor mechanism:** a linear correction head over `featurize(x,a)` vs a LightGBM custom
  objective vs a separate PyTorch model. (User chose the **linear head**.)
- **SPO+ "true prize":** the simulator oracle's best-action value vs the **realized logged reward**.
- **Head feature space:** raw features vs **standardized** features.
- **Decision prize corrected by SPO+:** the logged action's `q` alone vs the **argmax door value**
  `max_a q(x,a)` (what the router actually collects).
- **DF scope across the ensemble:** apply DF uniformly to all bootstrap members vs the **served
  point model only**.
- **Reweight boundary band:** value-collapsed prize quantiles vs **per-row rank** quantiles.
- **Leaderboard params:** a fresh scale vs matching the existing `baseline` row exactly.

## Reasoning

- **Linear head, fit in standardized space, folded back to raw.** A linear head over *raw* features
  (e.g. `property_value ~ 1e5`) makes the SPO+ subgradient explode — the first run drove
  `|head|_inf` to ~2.7e4 and *tripled* decision regret. LightGBM is scale-invariant, but a linear
  head is not. The shipped `spo_finetune` standardizes features (`(x-mu)/sigma`), steps `w` there,
  then folds the scale back into a raw-space head (`head = w/sigma`); the dropped constant offset
  shifts every score equally, so it changes no include/skip/argmax decision and is reabsorbed by the
  refit calibrator. `_predict` stays a plain `booster.predict(x) + x @ head`. `head=None` (default)
  and `spo_epochs=0`/`spo_lr=0` both reproduce the base model exactly (unit-tested).
- **Realized logged reward as the SPO+ prize (oracle-free).** SPO+ needs a "true cost" vector; the
  production-faithful, guard-legal choice is `event.reward`, not `true_reward`/`true_best_action`.
  This is the label the method is designed for and keeps `reward/decision_focused.py` clean under
  `tests/test_ethics.py::test_no_oracle_leak`.
- **Correct the argmax door value, not just the logged action.** The router prices a door by its
  best-action value `max_a q(x,a)`, so SPO+ steps the head through the argmax action per door (a
  proper subgradient of the max). Correcting only the logged action mis-aligned training and serving
  and made regret worse.
- **DF is a point-model concern; the ensemble stays plain.** `BootstrapEnsemble.fit` refits members
  with `use_decision_focused=False`. The ensemble exists to quantify *uncertainty* (spread) for
  Thompson/risk pricing; the DF correction is about the *point* decision, and at serve time
  (`door_profit`, risk off) the point model is what prices doors. This preserves the ensemble's
  uncertainty semantics and avoids an `n_bootstrap`-fold SPO+ cost. When the flag is off this is a
  no-op.
- **Synthetic walkable geometry for the SPO+ inner routes.** The logged simulator scatters doors
  over kilometres, so travel dominates and the include decision ignores prize — no learning signal.
  `_dense_coords` repositions each neighborhood onto a walkable block (exactly what
  `run_demo._dense_block` does for the graded shift), making prize and travel comparable. Geometry
  is safe to synthesize: `lat`/`lon` never enter a model (ethics allow-list).
- **Per-row rank band for reweighting.** On the discrete reward scale, collapsing tied prizes onto a
  single quantile can leave the central band empty; per-row rank quantiles guarantee ~`q` of rows
  (nearest the median = the boundary) are upweighted.
- **Match the baseline params** (6 shifts × seed 7, n_logs 3000, shift 40) so the rows compare to
  `baseline`/`phase10-*`/`phase11-*`.

## Leaderboard results (6 shifts × seed 7, n_logs 3000, shift 40, vs `baseline` +4.526)

| experiment | realized value | Δ value | decision regret | std | OPE LCB | verdict |
|---|---|---|---|---|---|---|
| baseline | +4.526 | +0.000 | 0.911 | 0.349 | +0.076 | neutral (reference) |
| phase12-reweight (`NBA_DF_MODE=reweight`) | +4.428 | -0.098 | **0.907** | 0.424 | **+0.077** | **regression** |
| phase12-spo (`NBA_DF_MODE=spo`) | +4.079 | -0.447 | 0.993 | 0.520 | +0.073 | **regression** |

- **Reweighting moves the right needle, just not the primary one.** It *lowers* decision regret
  (0.907 < 0.911) at *equal-or-better* OPE (LCB 0.077 > 0.076) — precisely the decision-focused
  claim — but realized shift value dips slightly (-0.098), so the primary-metric gate returns
  **regression**. As in Phases 10-11, the single dense-block demo does not exercise the value regime:
  the base LightGBM is already a near-optimal door ranker at this scale, so sharpening boundary
  decisions cannot lift realized value and the reweight's variance shows up as a small mean drop.
- **SPO+ regresses at single-block scale.** Realized value -0.447 and regret *up* (0.993): with
  single-sample bandit rewards as the "true prize", the SPO+ correction is high-variance and, against
  an already-strong base, does more harm than good here. It is mechanically correct (unit-tested:
  zero-step identity, non-zero learned head, still a calibrated `QModel`, save/load round-trip) but
  its value regime is a **weak-base / genuinely-capacity-bound** setting, shown in the notebook.
- Adoption stays **off by default** (both regressions correctly block default-on).

## Where the value actually shows (mechanics + regime)

- **Unit tests** (`tests/test_decision_focused.py`) prove the mechanics: default-off byte-identical;
  reweight upweights the boundary band and changes the fit; SPO+ zero-step identity; SPO+ is a
  non-zero, calibrated `QModel` that saves/loads; a DF model runs through the same `PromotionGate`;
  and, in a **weak-base regime** (300 logs, 16-door neighborhoods), reweighting **lowers exact
  top-capacity selection regret** while calibration survives.
- **Notebook** (`notebooks/decision_focused_demo.ipynb`) teaches the predict-then-optimize mismatch
  from zero, shows both on-ramps on the real pipeline, and plots the weak-base regret win, then
  cross-references these leaderboard rows so the story matches the recorded verdicts.

## Trade-offs accepted

- Like Phases 10-11, the single-block board shows the value story as a regression, so the **unit
  tests + notebook** (not the board) are the proof of correct mechanics and the value regime; the
  flag stays opt-in.
- SPO+ trains its prize on the logged action's realized reward (a single-sample estimate of the
  door's expected value) while the router prices by the argmax; the shared linear head over
  `featurize(x,a)` corrects the full `q` vector, but the label noise is the source of SPO+'s
  variance. A denoised prize (e.g. RDL residuals) is future work (Phase 16).
- The head correction is linear; a nonlinear decision-focused head is deferred.

## Supersedes

None. First Phase 12 decision; builds on the Phase 3/5 reward model + calibration, the Phase 6/10
`solve_tsp_profits` router, and the Phase 17 leaderboard.
