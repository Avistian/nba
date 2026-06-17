# NBA Core — Rules

Confirmed patterns — apply by default on core/architecture work.

## R1: Oracle isolation is non-negotiable

Learning modules (`reward`, `bandits`, `ope`, `routing`, `api`, `pipeline`) must never import
`true_reward`, `latent_scores`, `true_best_action`, or `outcome_probs`. Oracle is eval-only
(simulator, scripts, notebooks, tests). Enforced by `tests/test_ethics.py::test_no_oracle_leak`.

*Confirmed: Phase 0–8, AST scan in CI.*

## R2: Log propensity on every recommend

`Orchestrator.recommend` and `EventStore.append_decision` require `propensity > 0`. No retrofitting
`p` after the fact.

*Confirmed: Phase 7 store tests, Phase 8 e2e propensity test.*

## R3: Append-only event log

Never add `UPDATE`/`DELETE` to the event store schema. Outcome corrections are new rows.

*Confirmed: `tests/test_store.py`, ARCHITECTURE.md.*

## R4: Inject dependencies; use `build_app` for API tests

Orchestrator takes policy/model/engine/store/settings via constructor. FastAPI uses
`build_app(orchestrator)` so `TestClient` never touches production disk.

*Confirmed: `tests/test_api.py`, `tests/test_e2e.py`.*

## R5: Single seed flows through Settings

Use `Settings.seed` and passed `rng` objects; don't call `np.random.seed()` globally in library code.

*Confirmed: `conftest.py`, deterministic test suite.*

## R6: Bandit-weighted profit for routing (default)

Price doors with Σ_a π(a|x)·q(x,a), not raw argmax q, unless explicitly toggling `argmax_profit`.

*Confirmed: Phase 7 orchestrator design, demo notebook comparison.*
