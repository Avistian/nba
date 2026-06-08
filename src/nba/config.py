"""Application configuration via environment-overridable settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # ethics
    cap_exploration_in_sensitive: bool = True

    def ensure_dirs(self) -> None:
        """Create all configured output directories. Idempotent."""
        for path in (self.data_dir, self.model_dir, self.db_path.parent):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` singleton."""
    return Settings()
