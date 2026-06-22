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

    # Phase 18 — drift monitoring + conditional retrain loop. All flags default OFF so the
    # serve/demo path is byte-identical to Phases 0-8/9/17. Setting ``use_drift_monitoring=1``
    # activates the monitor/retrain batch job; it never touches the hot serve path.
    use_drift_monitoring: bool = False
    # gate ``DriftSpec`` injection in log generation (demos/grading)
    use_simulated_drift: bool = False
    # cap on events in the reference slice (since last promote)
    monitor_reference_window: int = 20_000
    monitor_recent_window: int = 2_000  # recent events scored for drift
    # run the monitor after this many new labeled outcomes
    monitor_interval_events: int = 500
    # minimum new labeled rows before a scheduled/drift retrain
    retrain_min_new_events: int = 2_000
    # scheduled safety retrain if no drift signal but data is stale
    retrain_max_age_days: int = 30
    drift_reward_psi_threshold: float = 0.15  # reward PSI trigger
    drift_calibration_delta_threshold: float = 0.05  # calibration MAE increase trigger
    drift_calibration_absolute_max: float = 0.12  # recent MAE absolute ceiling trigger
    drift_feature_psi_threshold: float = 0.20  # feature PSI (max over allow-list) trigger
    drift_rolling_dr_drop_threshold: float = 0.03  # rolling DR drop trigger (absolute)
    drift_min_propensity_floor: float = 0.02  # overlap warning floor
    drift_min_ess_fraction: float = 0.05  # ESS/n warning floor
    # optional sample weights for fit; None=uniform
    retrain_time_decay_halflife_days: float | None = None
    monitoring_report_path: Path = Path("artifacts/monitoring/drift_reports.jsonl")
    retrain_audit_path: Path = Path("artifacts/monitoring/retrain_audit.jsonl")
    deployed_model_manifest: Path = Path("artifacts/models/deployed.json")

    # Optional read-only observability stack. Off by default so tests and CI never need Docker.
    use_monitoring_dashboard: bool = False  # when on, documents/starts the optional Grafana stack
    metrics_exporter_enabled: bool = False  # expose Prometheus /metrics from drift JSONL + rollups
    metrics_exporter_port: int = 9091
    metrics_refresh_seconds: int = 30  # how often the exporter re-reads artifacts between scrapes

    def ensure_dirs(self) -> None:
        """Create all configured output directories. Idempotent."""
        for path in (
            self.data_dir,
            self.model_dir,
            self.db_path.parent,
            self.monitoring_report_path.parent,
            self.retrain_audit_path.parent,
            self.deployed_model_manifest.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` singleton."""
    return Settings()
