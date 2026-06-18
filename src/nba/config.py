"""Application configuration via environment-overridable settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Twelve-factor settings; every field overridable via an ``NBA_*`` env var."""

    model_config = SettingsConfigDict(env_prefix="NBA_", env_file=".env", extra="ignore")

    # paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("artifacts/models")
    db_path: Path = Path("artifacts/events.db")

    # determinism
    seed: int = 7

    # dataset selection (Phase 9). Default reproduces today exactly: nothing relational
    # is imported on the hot path while ``dataset_mode == "flat"``.
    dataset_mode: Literal["flat", "relational"] = "flat"
    relational_data_dir: Path = Path("data/relational")
    n_households: int = 0  # 0 => derive from n (~1 household per 3 doors)
    neighbor_radius_km: float = 0.15  # two doors get a ``near`` edge within this radius
    history_len: int = 8  # max prior interactions retained per prospect
    competitor_density: float = 0.2  # fraction of blocks carrying a competitor-overlap edge
    relational_seed: int = 7  # independent seed so the relational world is reproducible

    # bandit knobs
    epsilon: float = 0.10
    ucb_c: float = 1.0
    n_bootstrap: int = 16
    softmax_temp: float = 0.25

    # routing knobs
    shift_capacity: int = 40
    walking_speed_kmh: float = 4.5
    lambda_travel: float = 1.0
    drop_scale: float = 1000.0
    time_window: tuple[int, int] = (16, 19)

    # ope / gate
    ope_min_lift: float = 0.0
    ope_z: float = 1.96

    # experiment leaderboard (Phase 17). Infra knobs; they do not alter the served loop.
    leaderboard_path: Path = Path("artifacts/leaderboard.jsonl")
    baseline_experiment_id: str = "baseline"
    eval_n_shifts: int = 50  # simulated shifts per experiment (variance/CVaR need repeats)
    eval_seeds: tuple[int, ...] = (7,)  # seeds swept per experiment for reproducible spread
    eval_cvar_alpha: float = 0.2  # worst-tail fraction for the CVaR (downside) metric

    # ethics
    cap_exploration_in_sensitive: bool = True
    sensitive_prior_interactions: int = 4  # >= this many prior contacts flags a door sensitive
    sensitive_exploration_ceiling: float = 0.05  # max non-greedy mass allowed in a sensitive door

    def ensure_dirs(self) -> None:
        """Create all configured output directories. Idempotent."""
        for path in (self.data_dir, self.model_dir, self.db_path.parent):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` singleton."""
    return Settings()
