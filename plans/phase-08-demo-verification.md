# Phase 8 — Demo + cross-module verification

**Depends on:** Phase 7. **Goal:** an end-to-end, offline demo that runs the whole loop for a
simulated shift and prints a comparison report, plus a verification suite that asserts the
system-level claims from `PLAN.md` (OPE matches OBP, bandit beats uniform, regret trends down,
TSP-P drops outliers, API roundtrip, propensity everywhere, ethics allow-list enforced).

## Files to create

```
scripts/run_demo.py
tests/test_e2e.py
tests/test_ethics.py
README.md            # usage / quickstart section (update from Phase 0 stub)
```

## `scripts/run_demo.py` — full-shift simulation

Pipeline (all seeded, all offline):

1. `generate_logs(n, seed)` → bootstrap a logged dataset from the simulator.
2. `RewardModel.fit(logs)` → calibrated `q(x,a)`; persist to `artifacts/models`.
3. Build all three policies (ε-greedy, UCB, Thompson) on the model/ensemble.
4. **OPE selection:** for each policy compute DR value on the held-out logs; `PromotionGate`
   picks the best policy that beats the logging baseline. Print the IPS/DM/DR table + decision.
5. **Simulate a shift** of `N` doors (fresh simulator contexts):
   - `Orchestrator.plan_route(contexts)` → initial walkable route.
   - Walk the route: for each visited door, `recommend` → simulate `outcome` via the simulator →
     `feedback`; every `replan_every` doors, `replan(remaining)`.
   - Track cumulative reward, and **regret** = Σ (`true_reward(ctx, true_best_action)` −
     `true_reward(ctx, chosen)`) using the oracle (eval-only).
6. **Baselines for comparison:** uniform-random policy and `ExploitationBaseline` over the same
   seeded contexts.
7. **Report** (stdout + `artifacts/demo_report.json`):
   - cumulative reward: chosen bandit vs uniform vs exploit-only.
   - regret curve (per-round cumulative regret; assert downward trend on a smoothed window).
   - routing: doors visited vs dropped, total walk time vs a naive "visit-all nearest-neighbor"
     route → walk-time saved.
   - OPE table and the gate's promotion decision.

CLI: `--n-logs 20000 --shift 60 --replan-every 10 --seed 7`.

## `tests/test_e2e.py` — system claims (small N, seeded, fast)

- **Bandit beats uniform:** selected policy cumulative reward > uniform baseline (margin, seeded).
- **Gate beats logging baseline:** chosen policy's DR lower bound > empirical logging value.
- **Regret trend:** linear fit slope of cumulative-regret-per-round is ≤ 0 (non-increasing trend)
  over the shift, on a smoothed window (`trend_window`).
- **Router drops outliers:** in the demo route, injected far low-profit doors appear in `dropped`.
- **Propensity everywhere:** every decision row in the store has `propensity > 0` after the shift.
- **API roundtrip (integration):** spin `build_app(orchestrator)` with `TestClient`, run a few
  `recommend→feedback`, then `/route`; assert 2xx and propensities present.

## `tests/test_ethics.py` — guardrails

- **Allow-list:** `FEATURE_NAMES` contains no protected attribute and no geo/address column;
  assert the model is trained only on `ALLOWED_FEATURES` (+ weather/action one-hot).
- **Sensitive-context exploration cap:** with `cap_exploration_in_sensitive=True`, the effective
  exploration probability in a flagged context is ≤ a configured ceiling (policy honors the flag).
- **No oracle leak (repo-wide):** AST/grep assertion that `nba.reward`, `nba.bandits`, `nba.ope`,
  `nba.routing`, `nba.api`, `nba.pipeline` never import `true_reward`/`latent_scores`.

## README usage section

- Quickstart: `make setup` → `python scripts/generate_logs.py` → `python scripts/train_reward.py`
  → `python scripts/evaluate_policy.py` → `make demo` → `make api`.
- One-paragraph architecture recap with the through-line diagram and the "bandit proposes, router
  disposes" framing; link to `plans/` and `docs/`.

## Acceptance (the PLAN.md verification matrix, all green)

- `make demo` runs fully offline and prints the comparison report + writes `demo_report.json`.
- `make test` green with coverage on all core modules.
- OPE estimators match OBP within tolerance (Phase 5 test).
- Bandit beats uniform baseline; cumulative regret trends down.
- TSP-P returns a walkable subset, drops far-flung outliers, respects time windows + capacity.
- API smoke: `recommend → feedback → route` roundtrip; **propensity present on every recommend**;
  event log append-only.
- Ethics: no protected attributes in features; exploration capped in sensitive contexts; no oracle
  leakage.
