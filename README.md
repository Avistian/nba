# NBA — Door-to-Door Next Best Action

An offline-first prototype of a field-sales recommender:

```
context x → reward model q(x,a) → bandit policy (explore) → per-door profit → TSP-P walkable route
```

Every recommendation logs its propensity `p`; logs feed an off-policy-evaluation (OPE) gate that
must pass before a policy is promoted. **The bandit proposes, the router disposes.**

See [PLAN.md](PLAN.md) for the high-level plan, [plans/](plans) for detailed per-phase
implementation specs, [ARCHITECTURE.md](ARCHITECTURE.md) for how it all fits together, and
[docs/](docs) for concepts and data sources.

## Quickstart

```bash
make setup     # uv sync (creates .venv, installs runtime + dev deps)
make check     # lint + type + test

# generate data + train the reward model
uv run python scripts/generate_logs.py --n 20000 --out data/logs.parquet
uv run python scripts/train_reward.py  --logs data/logs.parquet --out artifacts/models

# off-policy-evaluate a candidate policy and gate its promotion
uv run python scripts/evaluate_policy.py --policy ucb

# (optional) generate the relational dataset + grade it on the experiment leaderboard
uv run python scripts/generate_relational_logs.py --n 20000 --out data/relational
uv run python scripts/run_experiment.py --baseline-only
uv run python scripts/run_experiment.py --experiment-id phase09-relational --phase 09 --dataset relational

# (optional) grade risk-aware routing (Phase 11): mean − κ·std pricing; κ=0 is an exact no-op
uv run python scripts/run_experiment.py --experiment-id phase11-risk-kappa05 --phase 11 \
    --set NBA_USE_RISK_AWARE_ROUTING=1 NBA_RISK_KAPPA=0.5

# run the whole loop offline for one simulated shift + print the comparison report
make demo      # = uv run python scripts/run_demo.py  (writes artifacts/demo_report.json)

# serve the closed loop (recommend / feedback / route / health)
make api        # = uv run uvicorn nba.api.app:app --reload  → POST /recommend, /feedback, /route
```

`make demo` is the fastest way to see everything at once: it bootstraps logs, trains the reward
model, off-policy-evaluates ε-greedy / UCB / Thompson, gates the winner, then walks a simulated
shift — comparing the chosen bandit against uniform-random and exploit-only baselines, measuring
regret against the oracle, and reporting routing time saved versus a naive visit-all tour.

## Architecture in one line

```
context x → reward model q(x,a) → bandit policy (+ ethics cap) → OPE gate → per-door profit → TSP-P route
                     ▲                                                                            │
                     └──────────────── append-only event log (context, action, reward, p) ◄──────┘
```

**The bandit proposes, the router disposes**, and every decision logs its propensity `p` so the
OPE gate can safely vet the *next* policy before it ever reaches the field. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full picture, [plans/](plans) for per-phase specs, and
[docs/](docs) for concepts — including [docs/09-build-nba-from-scratch.md](docs/09-build-nba-from-scratch.md),
a from-zero walkthrough of the entire system.

## Status

The full end-to-end loop is implemented and verified: simulator → reward model → bandit policies →
OPE gate → TSP-with-profits router → orchestrator + FastAPI service over an append-only event store,
with ethics guardrails and a system-level verification suite.

| Phase | Area | State |
|------:|------|-------|
| 0 | Scaffold, config, tooling | done |
| 1 | Schema + reward map + featurize | done |
| 2 | D2D simulator + feature substrate | done |
| 3 | Reward model (LightGBM + isotonic) | done |
| 4 | Bandit policies (ε-greedy, UCB, Thompson) | done |
| 5 | OPE estimators + promotion gate | done |
| 6 | Routing / TSP-with-profits | done |
| 7 | Orchestrator + FastAPI + event store | done |
| 8 | Demo + end-to-end + ethics verification | done |
| 9 | Relational dataset (mirrors flat, `dataset_mode`) | done |
| 10 | Upgrade 1 — orienteering (budget / team / road) | done |
| 11 | Upgrade 3 — risk-aware routing | done |
| 12 | Upgrade 2 — decision-focused learning | planned |
| 13 | Upgrade 5 — dynamic / stochastic routing | planned |
| 14 | Relational Deep Learning value model | planned |
| 15 | Upgrade 4 — neural combinatorial optimization | deferred |
| 16 | Decision-focused RDL (research frontier) | deferred |
| 17 | Experiment leaderboard (lift/regression eval) | done |

## Roadmap & feature flags

Phases 9–16 extend the verified loop with the upgrades from
[docs/11](docs/11-improving-nba-spatio-relational-optimization.md) (optimizer side) and
[docs/12](docs/12-relational-deep-learning-mixin.md) (value side). Two principles govern all of them:

- **Every upgrade is a feature flag, off by default.** Each adds `NBA_*` settings whose defaults
  reproduce today's behavior exactly (e.g. `NBA_DATASET_MODE=flat`, `NBA_RISK_KAPPA=0.0`,
  `NBA_REWARD_MODEL_KIND=lightgbm`), so the verified pipeline is untouched until you opt in.
- **The dataset becomes relational first** (Phase 9, **built**), added as a *new* dataset that mirrors
  the flat simulator's `BanditEvent` stream — the prerequisite for the Relational Deep Learning value
  model, benchmarked head-to-head against LightGBM through the same OPE gate. The relational structure
  (households, neighbor/competitor edges, interaction histories) rides in **additive sidecar
  artifacts** (`data/relational/{households,edges}.parquet`, `graph.npz`) plus one optional non-model
  `household_id` column, so the `BanditEvent` data contract is **unchanged** and every existing learner
  consumes relational logs as-is.
- **Every upgrade is tested and proves itself on a logged leaderboard** (Phase 17, **built**). Built
  right after the relational dataset and before the upgrades (order: **9 → 17 → 18 → 10–16**), Phase 17 adds
  an append-only `artifacts/leaderboard.jsonl` (+ `leaderboard.md`): each flag config is scored against
  the baseline and judged a **lift, regression, or neutral**, where a lift requires both a higher
  realized shift value and clearing the DR gate, and a regression blocks adoption.
  `scripts/run_experiment.py` records and prints the board.

See [PLAN.md](PLAN.md) for the phase list, [plans/](plans) for per-phase specs
(`phase-09`…`phase-16`), and [docs/](docs) for the step-by-step build docs (`13`…`20`).

Notebooks in [notebooks/](notebooks) mirror each phase and are **parametrized**: a single
`DATASET_MODE = "flat" | "relational"` cell at the top switches each notebook between the flat and the
relational dataset (resolving the log path, the trained model, and the grading oracle). They cover
EDA, reward-model explainability, display calibration, bandit behavior, off-policy evaluation,
TSP-with-profits routing, the orchestrator/API loop, the
[end-to-end demo](notebooks/end_to_end_demo.ipynb), and a relational-structure EDA
(`relational_structure_eda.ipynb`). The original (pre-parametrization) notebooks are preserved
unchanged in [notebooks/old/](notebooks/old). Per-upgrade demos accompany the roadmap phases:
[`team_orienteering_demo.ipynb`](notebooks/team_orienteering_demo.ipynb) (Phase 10 — where a team
beats one optimized rep) and
[`risk_aware_routing_demo.ipynb`](notebooks/risk_aware_routing_demo.ipynb) (Phase 11 — the exact
`κ=0` no-op and the realized-value risk-return frontier).

## Toolchain

- Python `>=3.11` (developed on 3.12), [`uv`](https://docs.astral.sh/uv/) for dependency
  management, `ruff` (lint/format), `pyright` (types), `pytest` (+`pytest-cov`).
- Host note: developed on linux/aarch64; pinned `numpy<2.0` for `obp`/`lightgbm` wheel
  compatibility.
