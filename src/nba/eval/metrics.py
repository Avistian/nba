"""The common, comparable metric set computed per experiment (the doc 11 §10 yardsticks).

Metrics are computed from a set of simulated shifts using the simulator oracle **for grading only**
(never for serving). A leaderboard run is, by construction, a *graded* demo run: :func:`evaluate`
reuses :func:`run_demo` so the numbers are exactly what the end-to-end demo would report, aggregated
across seeds and repeated shifts for variance/CVaR.
"""

from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from nba.config import Settings

# run_demo lives under scripts/; make it importable the same way the e2e test does.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from run_demo import run_demo  # noqa: E402


@dataclass(frozen=True)
class ExperimentMetrics:
    """The metric set graded per experiment. ``realized_shift_value_mean`` is the PRIMARY metric."""

    realized_shift_value_mean: float  # total true reward captured / shift
    realized_shift_value_std: float  # spread across shifts/seeds
    realized_shift_value_cvar: float  # mean of the worst ``eval_cvar_alpha`` tail (downside)
    decision_regret_mean: float  # value lost vs the oracle that knew true prizes (U2)
    ope_value: float  # DR point estimate of the selected policy value
    ope_lcb: float  # DR lower confidence bound (the gate quantity)
    optimality_gap: float | None  # learned-router value / OR-Tools value (U4 only; None for now)
    route_time_s_mean: float  # operational cost (sanity)

    def to_dict(self) -> dict[str, float | None]:
        """Return a JSON-serializable mapping of the metrics."""
        return asdict(self)


def _cvar(values: np.ndarray, alpha: float) -> float:
    """Mean of the worst ``alpha`` fraction (lower tail) of ``values``."""
    if values.size == 0:
        return 0.0
    k = max(1, math.ceil(alpha * values.size))
    worst = np.sort(values)[:k]
    return float(worst.mean())


def evaluate(
    *,
    settings: Settings,
    n_shifts: int,
    seeds: tuple[int, ...],
    n_logs: int = 4_000,
    shift: int = 24,
    replan_every: int = 8,
) -> ExperimentMetrics:
    """Walk ``n_shifts`` simulated shifts (per seed) under ``settings``; aggregate the yardsticks.

    Each (seed, shift) pair is one full graded demo run via :func:`run_demo` under the experiment's
    flag config. Model/store artifacts are redirected to a throwaway directory so grading never
    touches the real ``artifacts/``.
    """
    realized: list[float] = []
    regrets: list[float] = []
    ope_values: list[float] = []
    ope_lcbs: list[float] = []
    route_times_s: list[float] = []

    with tempfile.TemporaryDirectory(prefix="nba-eval-") as tmp:
        tmp_path = Path(tmp)
        for seed in seeds:
            for k in range(n_shifts):
                run_seed = seed * 10_000 + k
                run_settings = settings.model_copy(
                    update={
                        "model_dir": tmp_path / "models",
                        "db_path": tmp_path / f"events-{run_seed}.db",
                    }
                )
                report = run_demo(
                    n_logs=n_logs,
                    shift=shift,
                    replan_every=replan_every,
                    seed=run_seed,
                    settings=run_settings,
                    ope_max_rows=min(800, n_logs),
                    write=False,
                )
                er = report.expected_reward
                realized.append(er["bandit"])
                regrets.append(er["oracle_best"] - er["bandit"])
                ope_values.append(report.ope[report.selected_policy]["dr"])
                ope_lcbs.append(report.selected_dr_lower_bound)
                route_times_s.append(report.routing["route_time_min"] * 60.0)

    realized_arr = np.asarray(realized, dtype=np.float64)
    return ExperimentMetrics(
        realized_shift_value_mean=float(realized_arr.mean()),
        realized_shift_value_std=float(realized_arr.std()),
        realized_shift_value_cvar=_cvar(realized_arr, settings.eval_cvar_alpha),
        decision_regret_mean=float(np.mean(regrets)),
        ope_value=float(np.mean(ope_values)),
        ope_lcb=float(np.mean(ope_lcbs)),
        optimality_gap=None,
        route_time_s_mean=float(np.mean(route_times_s)),
    )
