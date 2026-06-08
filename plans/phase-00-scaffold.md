# Phase 0 — Scaffold

**Depends on:** nothing. **Parallelizable with:** nothing (foundation).
**Goal:** a clean, installable package skeleton with pinned tooling so every later phase has a
home, a config object, and a green (empty) test run.

## Files to create

```
nba/
  pyproject.toml
  README.md
  Makefile
  .gitignore
  .python-version            # "3.11"
  src/nba/__init__.py        # __version__ = "0.1.0"
  src/nba/config.py
  src/nba/py.typed           # PEP 561 marker (ships types)
  tests/conftest.py
  tests/test_config.py
```

## `pyproject.toml`

- Build backend: `hatchling`. `[project]` name `nba`, `requires-python = ">=3.11"`.
- **Runtime deps** (pin minor): `lightgbm>=4.3`, `ortools>=9.10`, `obp>=0.5.7`, `fastapi>=0.111`,
  `uvicorn[standard]>=0.30`, `pydantic>=2.7`, `pydantic-settings>=2.3`, `pandas>=2.2`,
  `numpy>=1.26,<2.0` (obp/lightgbm wheel compat — verify), `scikit-learn>=1.5`, `scipy>=1.13`,
  `joblib>=1.4`, `pyarrow>=16` (parquet).
- **Dev deps** (`[dependency-groups]` or `[project.optional-dependencies].dev`): `pytest>=8.2`,
  `pytest-cov>=5`, `ruff>=0.5`, `pyright>=1.1.370`, `httpx>=0.27` (FastAPI TestClient).
- `[tool.ruff]`: `line-length = 100`, select `E,F,I,UP,B,SIM`, target `py311`.
- `[tool.pyright]`: `typeCheckingMode = "standard"`, `pythonVersion = "3.11"`, include `src`,`tests`.
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `addopts = "-q --strict-markers"`,
  marker `slow` (network/OBD downloads) registered.
- `[tool.coverage.run]`: `source = ["nba"]`, `branch = true`.

## `src/nba/config.py`

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NBA_", env_file=".env")

    # paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts/models")
    db_path: Path = Path("artifacts/events.db")
    # determinism
    seed: int = 7
    # bandit knobs
    epsilon: float = 0.10
    ucb_c: float = 1.0
    n_bootstrap: int = 16          # Thompson ensemble size
    softmax_temp: float = 0.25     # UCB action_dist smoothing
    # routing knobs
    shift_capacity: int = 40       # max doors per route
    walking_speed_kmh: float = 4.5
    lambda_travel: float = 1.0     # travel-time weight vs profit
    drop_scale: float = 1000.0     # profit→penalty integer scale for AddDisjunction
    time_window: tuple[int, int] = (16, 19)  # residential knock hours
    # ope / gate
    ope_min_lift: float = 0.0      # required lift over logging baseline
    ope_z: float = 1.96            # one-sided CI multiplier
    # ethics
    cap_exploration_in_sensitive: bool = True

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.model_dir, self.db_path.parent):
            p.mkdir(parents=True, exist_ok=True)

def get_settings() -> Settings: ...   # lru_cache singleton
```

- All knobs overridable via `NBA_*` env vars (twelve-factor). `ensure_dirs()` is idempotent.

## `Makefile`

| Target | Command |
|--------|---------|
| `setup` | `uv sync --all-extras` |
| `lint`  | `uv run ruff check . && uv run ruff format --check .` |
| `fmt`   | `uv run ruff format .` |
| `type`  | `uv run pyright` |
| `test`  | `uv run pytest --cov=nba --cov-report=term-missing` |
| `demo`  | `uv run python scripts/run_demo.py` |
| `api`   | `uv run uvicorn nba.api.app:app --reload` |
| `check` | `make lint type test` |

## `.gitignore`

`__pycache__/`, `.venv/`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`,
`htmlcov/`, `data/*.parquet`, `artifacts/`, `.env`.

## Tests

- `tests/conftest.py`: fixtures `settings` (tmp-path-backed `Settings`), `rng`
  (`np.random.default_rng(0)`), `tmp_model_dir`. Autouse fixture sets `NBA_SEED=7`.
- `tests/test_config.py`:
  - defaults load; `ensure_dirs()` creates all dirs and is idempotent on second call.
  - env override: monkeypatch `NBA_EPSILON=0.3` → `get_settings()` (cache cleared) reflects it.
  - `get_settings()` returns the same cached instance.

## Acceptance

- `uv sync` resolves; `uv run python -c "import nba; print(nba.__version__)"` prints `0.1.0`.
- `uv run pytest` collects ≥3 tests, all green.
- `uv run ruff check .` and `uv run pyright` report zero issues.
- Wheels for `lightgbm`, `ortools`, `obp` install on Python 3.11 (record exact resolved versions
  in `README.md`); if `numpy 2.x` breaks obp, pin `<2.0` and note why.

## Risks / notes

- `obp` historically lags numpy majors — resolve the numpy pin here so later phases are stable.
- Keep `config.py` dependency-free (no project imports) to avoid import cycles.
