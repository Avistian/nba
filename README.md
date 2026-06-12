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

Notebooks in [notebooks/](notebooks) mirror each phase: EDA, reward-model explainability, display
calibration, bandit behavior, off-policy evaluation, TSP-with-profits routing, the orchestrator/API
loop, and the [end-to-end demo](notebooks/end_to_end_demo.ipynb).

## Toolchain

- Python `>=3.11` (developed on 3.12), [`uv`](https://docs.astral.sh/uv/) for dependency
  management, `ruff` (lint/format), `pyright` (types), `pytest` (+`pytest-cov`).
- Host note: developed on linux/aarch64; pinned `numpy<2.0` for `obp`/`lightgbm` wheel
  compatibility.
