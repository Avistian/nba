# NBA — Door-to-Door Next Best Action

An offline-first prototype of a field-sales recommender:

```
context x → reward model q(x,a) → bandit policy (explore) → per-door profit → TSP-P walkable route
```

Every recommendation logs its propensity `p`; logs feed an off-policy-evaluation (OPE) gate that
must pass before a policy is promoted. **The bandit proposes, the router disposes.**

See [PLAN.md](PLAN.md) for the high-level plan, [plans/](plans) for detailed per-phase
implementation specs, and [docs/](docs) for concepts and data sources.

## Quickstart

```bash
make setup     # uv sync (creates .venv, installs runtime + dev deps)
make test      # pytest with coverage
make check     # lint + type + test
```

## Status

- **Phase 0 — Scaffold:** done (package skeleton, config, tooling, tests).

## Toolchain

- Python `>=3.11` (developed on 3.12), [`uv`](https://docs.astral.sh/uv/) for dependency
  management, `ruff` (lint/format), `pyright` (types), `pytest` (+`pytest-cov`).
- Host note: developed on linux/aarch64; pinned `numpy<2.0` for `obp`/`lightgbm` wheel
  compatibility.
