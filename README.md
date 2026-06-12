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

# serve the closed loop (recommend / feedback / route / health)
uv run uvicorn nba.api.app:app --reload   # then POST /recommend, /feedback, /route
```

## Status

The end-to-end loop is implemented: simulator → reward model → bandit policies → OPE gate →
TSP-with-profits router → orchestrator + FastAPI service over an append-only event store.

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
| 8 | Demo + end-to-end verification | planned |

Notebooks in [notebooks/](notebooks) mirror each phase: EDA, reward-model explainability, display
calibration, bandit behavior, off-policy evaluation, TSP-with-profits routing, and the
orchestrator/API loop.

## Toolchain

- Python `>=3.11` (developed on 3.12), [`uv`](https://docs.astral.sh/uv/) for dependency
  management, `ruff` (lint/format), `pyright` (types), `pytest` (+`pytest-cov`).
- Host note: developed on linux/aarch64; pinned `numpy<2.0` for `obp`/`lightgbm` wheel
  compatibility.
