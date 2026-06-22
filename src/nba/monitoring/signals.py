"""Drift signals — score reference vs recent logged batches.

Every signal is computed **offline** on logged ``(context, action, reward,
propensity)`` — no oracle, no protected/geo fields. Each returns a
:class:`DriftSignal` carrying a scalar value, its threshold, a pass/fail flag,
and a human-readable detail string.

PSI bin edges are **fixed** from the reference window (or stored in the deployed
manifest) so scores stay comparable across monitor runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from nba.bandits.base import Policy
from nba.config import Settings
from nba.data.features import ALLOWED_FEATURES, WEATHER_LEVELS, featurize
from nba.ope.estimators import LoggedBatch, dr, eval_action_matrix, q_matrix
from nba.reward.model import RewardModel
from nba.schema import ACTIONS, Outcome

#: Reward-ladder edges for PSI on the realized reward distribution. Mirrors the
#: REWARD map values; fixed across runs so values stay comparable.
REWARD_PSI_BINS: np.ndarray = np.array(sorted({-0.21, 0.0, 0.1, 0.3, 1.0, 1.01}), dtype=np.float64)


@dataclass(frozen=True)
class DriftSignal:
    """One drift signal's score, threshold, pass/fail flag, and detail."""

    name: str
    value: float
    threshold: float
    triggered: bool
    detail: str


@dataclass(frozen=True)
class DriftReport:
    """The full scored drift report — append-only to ``drift_reports.jsonl``."""

    timestamp: datetime
    n_reference: int
    n_recent: int
    signals: tuple[DriftSignal, ...]
    overlap_ok: bool

    def signal(self, name: str) -> DriftSignal:
        """Return the signal named ``name`` (raises if missing)."""
        for s in self.signals:
            if s.name == name:
                return s
        raise KeyError(f"no signal named {name!r}")

    def to_json(self) -> dict[str, object]:
        """Return a JSON-serializable mapping for one append-only line."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "n_reference": self.n_reference,
            "n_recent": self.n_recent,
            "overlap_ok": self.overlap_ok,
            "signals": [
                {
                    "name": s.name,
                    "value": s.value,
                    "threshold": s.threshold,
                    "triggered": s.triggered,
                    "detail": s.detail,
                }
                for s in self.signals
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> DriftReport:
        """Reconstruct a report from one parsed JSONL line."""
        return cls(
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
            n_reference=int(data["n_reference"]),  # type: ignore[arg-type]
            n_recent=int(data["n_recent"]),  # type: ignore[arg-type]
            overlap_ok=bool(data["overlap_ok"]),
            signals=tuple(
                DriftSignal(
                    name=str(s["name"]),  # type: ignore[index]
                    value=float(s["value"]),  # type: ignore[index]
                    threshold=float(s["threshold"]),  # type: ignore[index]
                    triggered=bool(s["triggered"]),  # type: ignore[index]
                    detail=str(s["detail"]),  # type: ignore[index]
                )
                for s in data["signals"]  # type: ignore[index]
            ),
        )


# --------------------------------------------------------------------------------------------- #
# PSI helpers
# --------------------------------------------------------------------------------------------- #
def _as_fractions(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Histogram ``values`` onto ``bins`` and return the fraction per bucket.

    ``bins`` are right-open edges ``[b0, b1, ..., bk]``; values < b0 fall in the
    first bucket, values >= b_{k-1} in the last. The returned vector sums to 1.
    """
    if values.size == 0:
        return np.zeros(len(bins) - 1, dtype=np.float64)
    idx = np.clip(np.searchsorted(bins[1:-1], values, side="right"), 0, len(bins) - 2)
    counts = np.bincount(idx, minlength=len(bins) - 1).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return np.zeros(len(bins) - 1, dtype=np.float64)
    return counts / total


def population_stability_index(
    reference: np.ndarray, cur: np.ndarray, *, bins: np.ndarray, eps: float = 1e-6
) -> float:
    """Return the PSI between ``reference`` and ``cur`` over ``bins``.

    PSI = Σ (cur% − ref%) · ln(cur% / ref%), with a tiny floor to avoid log(0).
    Returns 0.0 on identical distributions.
    """
    ref_p = _as_fractions(np.asarray(reference, dtype=np.float64), bins)
    cur_p = _as_fractions(np.asarray(cur, dtype=np.float64), bins)
    ref_p = np.maximum(ref_p, eps)
    cur_p = np.maximum(cur_p, eps)
    diff = cur_p - ref_p
    ratio = cur_p / ref_p
    return float(np.sum(diff * np.log(ratio)))


# --------------------------------------------------------------------------------------------- #
# Per-signal scorers
# --------------------------------------------------------------------------------------------- #
def reward_psi(reference: LoggedBatch, recent: LoggedBatch, *, settings: Settings) -> DriftSignal:
    """PSI between reference and recent reward distributions on the fixed reward ladder."""
    threshold = settings.drift_reward_psi_threshold
    value = population_stability_index(reference.rewards, recent.rewards, bins=REWARD_PSI_BINS)
    triggered = value > threshold
    return DriftSignal(
        name="reward_psi",
        value=value,
        threshold=threshold,
        triggered=triggered,
        detail=f"reward PSI={value:.4f} (>{threshold:.3f})",
    )


def calibration_mae(model: RewardModel, batch: LoggedBatch, *, settings: Settings) -> float:
    """Mean |q(x, a_logged) − r| on a labeled batch."""
    del settings  # unused; kept for signature symmetry
    q_logged = np.array(
        [
            model.q(ctx, ACTIONS[int(a)])
            for ctx, a in zip(batch.contexts, batch.actions, strict=True)
        ],
        dtype=np.float64,
    )
    return float(np.mean(np.abs(q_logged - batch.rewards)))


def calibration_drift(
    model: RewardModel,
    reference: LoggedBatch,
    recent: LoggedBatch,
    *,
    settings: Settings,
) -> DriftSignal:
    """Δ calibration MAE (recent − reference) plus an absolute ceiling trigger."""
    delta_threshold = settings.drift_calibration_delta_threshold
    abs_max = settings.drift_calibration_absolute_max
    ref_mae = calibration_mae(model, reference, settings=settings)
    recent_mae = calibration_mae(model, recent, settings=settings)
    delta = recent_mae - ref_mae
    triggered = (delta > delta_threshold) or (recent_mae > abs_max)
    detail = f"calib Δ={delta:+.4f} (> {delta_threshold:.3f}) | recent_mae={recent_mae:.4f}"
    return DriftSignal(
        name="calibration_drift",
        value=delta,
        threshold=delta_threshold,
        triggered=triggered,
        detail=detail,
    )


def _context_matrix(contexts: Sequence) -> np.ndarray:
    """Allow-list numeric/bool + weather one-hot for a list of contexts."""
    return np.vstack(
        [
            featurize(ctx, ACTIONS[0])[: len(ALLOWED_FEATURES) + len(WEATHER_LEVELS)]
            for ctx in contexts
        ]
    )


def feature_psi_max(
    reference: LoggedBatch, recent: LoggedBatch, *, settings: Settings
) -> DriftSignal:
    """Max PSI over allow-listed context features (geo/identity never enter)."""
    threshold = settings.drift_feature_psi_threshold
    ref_x = _context_matrix(reference.contexts)
    cur_x = _context_matrix(recent.contexts)
    # Per-column empirical bins: 10 quantile edges from the reference, floored at min.
    n_cols = ref_x.shape[1]
    max_psi = 0.0
    worst_col = -1
    for j in range(n_cols):
        col_ref = ref_x[:, j]
        col_cur = cur_x[:, j]
        # Drop NaNs (none expected; defensive).
        col_ref = col_ref[~np.isnan(col_ref)]
        col_cur = col_cur[~np.isnan(col_cur)]
        if col_ref.size == 0 or col_cur.size == 0:
            continue
        # Fixed 10-quantile bins from reference.
        qs = np.quantile(col_ref, np.linspace(0.0, 1.0, 11))
        qs = np.unique(qs)
        if qs.size < 2:
            continue  # constant column → no PSI contribution
        psi = population_stability_index(col_ref, col_cur, bins=qs)
        if psi > max_psi:
            max_psi = psi
            worst_col = j
    name = (
        (list(ALLOWED_FEATURES) + [f"weather={w}" for w in WEATHER_LEVELS])[worst_col]
        if worst_col >= 0
        else "<none>"
    )
    triggered = max_psi > threshold
    return DriftSignal(
        name="feature_psi_max",
        value=max_psi,
        threshold=threshold,
        triggered=triggered,
        detail=f"max feature PSI={max_psi:.4f} on {name} (>{threshold:.3f})",
    )


def _effective_sample_size(weights: np.ndarray) -> float:
    """ESS = (Σw)² / Σw² — the IPS-equivalent sample size."""
    denom = float(np.sum(weights**2))
    if denom <= 0.0:
        return 0.0
    return float(weights.sum() ** 2 / denom)


def overlap_health(recent: LoggedBatch, *, settings: Settings) -> DriftSignal:
    """Overlap health on the recent batch — min propensity and ESS/n."""
    min_p = float(np.min(recent.propensities))
    ess = _effective_sample_size(np.reciprocal(np.maximum(recent.propensities, 1e-12)))
    ess_frac = ess / len(recent)
    floor_p = settings.drift_min_propensity_floor
    floor_ess = settings.drift_min_ess_fraction
    ok = (min_p >= floor_p) and (ess_frac >= floor_ess)
    value = min(min_p, ess_frac)  # single scalar: the worse of the two
    threshold = min(floor_p, floor_ess)
    return DriftSignal(
        name="overlap_health",
        value=value,
        threshold=threshold,
        triggered=not ok,
        detail=(
            f"min_p={min_p:.4f} (floor {floor_p:.3f}) | "
            f"ess/n={ess_frac:.4f} (floor {floor_ess:.3f})"
        ),
    )


def rolling_dr_drop(
    model: RewardModel,
    policy: Policy,
    recent: LoggedBatch,
    *,
    deployed_dr: float | None,
    settings: Settings,
) -> DriftSignal:
    """DR estimate of the deployed policy on the recent window, vs DR at last promotion.

    When ``deployed_dr`` is unknown (first run / no manifest yet), the signal is
    a no-op (value 0, not triggered).
    """
    threshold = settings.drift_rolling_dr_drop_threshold
    if deployed_dr is None:
        return DriftSignal(
            name="rolling_dr_drop",
            value=0.0,
            threshold=threshold,
            triggered=False,
            detail="no deployed DR baseline (first run)",
        )
    q_hat = q_matrix(model, recent.contexts)
    pi_e = eval_action_matrix(policy, recent.contexts)
    recent_dr = dr(recent, q_hat, pi_e).value
    drop = deployed_dr - recent_dr  # positive => recent worse than deployed
    triggered = drop > threshold
    return DriftSignal(
        name="rolling_dr_drop",
        value=float(drop),
        threshold=threshold,
        triggered=triggered,
        detail=f"recent_dr={recent_dr:.4f} vs deployed={deployed_dr:.4f} (drop>{threshold:.3f})",
    )


# --------------------------------------------------------------------------------------------- #
# Report builder
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DriftReportContext:
    """Inputs for :func:`build_drift_report` — model + policy + windows."""

    model: RewardModel
    policy: Policy
    reference: LoggedBatch
    recent: LoggedBatch
    deployed_dr: float | None = None


def build_drift_report(
    *,
    ctx: DriftReportContext,
    settings: Settings,
    now: datetime | None = None,
) -> DriftReport:
    """Build a full :class:`DriftReport` from the five per-signal scorers."""
    signals: list[DriftSignal] = [
        reward_psi(ctx.reference, ctx.recent, settings=settings),
        calibration_drift(ctx.model, ctx.reference, ctx.recent, settings=settings),
        feature_psi_max(ctx.reference, ctx.recent, settings=settings),
        overlap_health(ctx.recent, settings=settings),
        rolling_dr_drop(
            ctx.model,
            ctx.policy,
            ctx.recent,
            deployed_dr=ctx.deployed_dr,
            settings=settings,
        ),
    ]
    overlap_ok = not signals[3].triggered  # overlap_health
    return DriftReport(
        timestamp=now or datetime.now(UTC),
        n_reference=len(ctx.reference),
        n_recent=len(ctx.recent),
        signals=tuple(signals),
        overlap_ok=overlap_ok,
    )


def append_report(report: DriftReport, path) -> None:
    """Append one report line to ``drift_reports.jsonl`` (creates parents)."""
    from pathlib import Path  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_json_dumps(report.to_json()) + "\n")


def _json_dumps(obj: dict[str, object]) -> str:
    """JSON dump that handles ``datetime`` and ``np.*`` scalar types."""
    import json  # noqa: PLC0415

    return json.dumps(obj, default=_json_default)


def _json_default(obj: object) -> object:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


# Re-export for convenience.
__all__ = [
    "DriftReport",
    "DriftReportContext",
    "DriftSignal",
    "REWARD_PSI_BINS",
    "append_report",
    "build_drift_report",
    "calibration_drift",
    "calibration_mae",
    "feature_psi_max",
    "overlap_health",
    "population_stability_index",
    "reward_psi",
    "rolling_dr_drop",
]


# Silence unused-import warnings for re-exports the public API keeps.
_OUTCOME_REF: Outcome = Outcome.NOT_HOME
