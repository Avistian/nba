"""Off-policy value estimators: IPS, SNIPS, DM, and DR.

Given logs collected under a *logging* policy (each row carries its propensity `p`), estimate the
value `V(π_e) = E_x E_{a∼π_e(·|x)}[ r ]` of a different *target* policy `π_e` — **without** running
it in the field. Three families trade bias against variance:

- **IPS / SNIPS** — importance-weight logged rewards by `π_e(a|x)/p`. Unbiased (given overlap) but
  high variance when weights are large; SNIPS self-normalizes to cut variance.
- **DM** — average the reward model's `q̂(x,a)` under `π_e`. Low variance, but biased if `q̂` is off.
- **DR** — DM baseline + IPS correction on the residual `r − q̂(x,a)`. Unbiased if *either* `q̂` or
  the propensities are right ("doubly robust"), and lower variance than plain IPS.

Nothing here imports the simulator oracle: these see only logged `(context, action, reward, p)`.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nba.bandits.base import Policy
from nba.reward.model import RewardModel
from nba.schema import ACTIONS, ProspectContext

_ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
_N_ACTIONS = len(ACTIONS)


@dataclass(frozen=True)
class LoggedBatch:
    """Logged feedback in array form: int-encoded actions, rewards, and logging propensities."""

    contexts: list[ProspectContext]
    actions: np.ndarray  # int-encoded over ACTIONS, shape (n,)
    rewards: np.ndarray  # shape (n,)
    propensities: np.ndarray  # logging p, shape (n,), all > 0

    def __post_init__(self) -> None:
        n = len(self.contexts)
        if not (len(self.actions) == len(self.rewards) == len(self.propensities) == n):
            raise ValueError("contexts/actions/rewards/propensities length mismatch")
        if n == 0:
            raise ValueError("empty batch")
        if np.any(self.propensities <= 0.0):
            raise ValueError("all propensities must be > 0 (overlap requirement)")

    def __len__(self) -> int:
        return len(self.contexts)

    @classmethod
    def from_events(cls, events: Sequence) -> LoggedBatch:
        """Build a batch from labeled :class:`~nba.schema.BanditEvent`s (drops unlabeled rows)."""
        labeled = [e for e in events if e.reward is not None]
        if not labeled:
            raise ValueError("no labeled events (reward is None for all)")
        return cls(
            contexts=[e.context for e in labeled],
            actions=np.array([_ACTION_INDEX[e.action] for e in labeled], dtype=np.int64),
            rewards=np.array([e.reward for e in labeled], dtype=np.float64),
            propensities=np.array([e.propensity for e in labeled], dtype=np.float64),
        )


@dataclass(frozen=True)
class OPEResult:
    """A single estimator's value, its standard error, and the sample size."""

    estimator: str
    value: float
    std_err: float
    n: int

    def ci(self, z: float) -> tuple[float, float]:
        """Return the ``z``-sigma confidence interval ``(value - z·se, value + z·se)``."""
        return (self.value - z * self.std_err, self.value + z * self.std_err)


def eval_action_matrix(policy: Policy, contexts: Sequence[ProspectContext]) -> np.ndarray:
    """Return the target policy's ``(n, |A|)`` probability matrix ``π_e(a | x)``."""
    rows = [[policy.action_dist(ctx)[a] for a in ACTIONS] for ctx in contexts]
    return _validate_pi(np.asarray(rows, dtype=np.float64), len(contexts))


def q_matrix(model: RewardModel, contexts: Sequence[ProspectContext]) -> np.ndarray:
    """Return the reward model's ``(n, |A|)`` matrix ``q̂(x, a)``."""
    if not contexts:
        raise ValueError("empty contexts")
    return np.vstack([model.q_all(ctx) for ctx in contexts])


# --------------------------------------------------------------------------------------------- #
# Internal guards / helpers
# --------------------------------------------------------------------------------------------- #
def _validate_pi(pi_e: np.ndarray, n: int, *, tol: float = 1e-6) -> np.ndarray:
    pi = np.asarray(pi_e, dtype=np.float64)
    if pi.shape != (n, _N_ACTIONS):
        raise ValueError(f"pi_e must have shape ({n}, {_N_ACTIONS}), got {pi.shape}")
    if np.any(pi < -tol):
        raise ValueError("pi_e has negative probabilities")
    if not np.allclose(pi.sum(axis=1), 1.0, atol=tol):
        raise ValueError("pi_e rows must sum to 1")
    return pi


def _validate_q(q_hat: np.ndarray, n: int) -> np.ndarray:
    q = np.asarray(q_hat, dtype=np.float64)
    if q.shape != (n, _N_ACTIONS):
        raise ValueError(f"q_hat must have shape ({n}, {_N_ACTIONS}), got {q.shape}")
    return q


def _weights(batch: LoggedBatch, pi_e: np.ndarray, clip: float | None) -> np.ndarray:
    """Importance weights ``π_e(a_i|x_i) / p_i``, optionally clipped to cap variance."""
    rows = np.arange(len(batch))
    w = pi_e[rows, batch.actions] / batch.propensities
    if clip is not None:
        w = np.minimum(w, clip)
    _warn_low_ess(w)
    return w


def _warn_low_ess(w: np.ndarray) -> None:
    """Warn if effective sample size ``(Σw)²/Σw²`` is a tiny fraction of ``n`` (poor overlap)."""
    denom = float(np.sum(w**2))
    if denom <= 0.0:
        return
    ess = float(w.sum() ** 2 / denom)
    if ess < 0.1 * len(w):
        warnings.warn(
            f"low effective sample size {ess:.1f}/{len(w)} — weights are concentrated "
            "(overlap problem); estimates may be unreliable",
            stacklevel=3,
        )


def _se(per_row: np.ndarray) -> float:
    """Closed-form standard error of a mean: ``std(per_row, ddof=1) / sqrt(n)``."""
    n = len(per_row)
    if n < 2:
        return 0.0
    return float(np.std(per_row, ddof=1) / np.sqrt(n))


def _bootstrap_se(per_row: np.ndarray, n_boot: int, rng: np.random.Generator) -> float:
    """Bootstrap standard error of the mean of ``per_row`` over ``n_boot`` resamples."""
    n = len(per_row)
    if n < 2:
        return 0.0
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        means[b] = per_row[rng.integers(0, n, size=n)].mean()
    return float(np.std(means, ddof=1))


# --------------------------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------------------------- #
def ips(batch: LoggedBatch, pi_e: np.ndarray, *, clip: float | None = None) -> OPEResult:
    """Inverse-propensity-scoring value: ``V = mean(w_i · r_i)``."""
    pi_e = _validate_pi(pi_e, len(batch))
    w = _weights(batch, pi_e, clip)
    per_row = w * batch.rewards
    return OPEResult("ips", float(per_row.mean()), _se(per_row), len(batch))


def snips(batch: LoggedBatch, pi_e: np.ndarray, *, clip: float | None = None) -> OPEResult:
    """Self-normalized IPS: ``V = Σ w_i r_i / Σ w_i`` (lower variance, slight bias)."""
    pi_e = _validate_pi(pi_e, len(batch))
    w = _weights(batch, pi_e, clip)
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return OPEResult("snips", 0.0, 0.0, len(batch))
    value = float(np.sum(w * batch.rewards) / w_sum)
    # Influence-function se for a ratio estimator: w_i (r_i − V) / mean(w).
    mean_w = w_sum / len(batch)
    per_row = w * (batch.rewards - value) / mean_w
    return OPEResult("snips", value, _se(per_row), len(batch))


def dm(
    batch: LoggedBatch,
    q_hat: np.ndarray,
    pi_e: np.ndarray,
    *,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> OPEResult:
    """Direct method: ``V = mean_i Σ_a π_e(a|x_i) q̂(x_i,a)`` (se via row bootstrap)."""
    pi_e = _validate_pi(pi_e, len(batch))
    q_hat = _validate_q(q_hat, len(batch))
    per_row = np.sum(pi_e * q_hat, axis=1)
    rng = rng if rng is not None else np.random.default_rng(0)
    return OPEResult("dm", float(per_row.mean()), _bootstrap_se(per_row, n_boot, rng), len(batch))


def dr(
    batch: LoggedBatch, q_hat: np.ndarray, pi_e: np.ndarray, *, clip: float | None = None
) -> OPEResult:
    """Doubly-robust value: DM baseline + IPS correction on the residual ``r − q̂(x,a)``."""
    pi_e = _validate_pi(pi_e, len(batch))
    q_hat = _validate_q(q_hat, len(batch))
    rows = np.arange(len(batch))
    baseline = np.sum(pi_e * q_hat, axis=1)
    q_chosen = q_hat[rows, batch.actions]
    w = _weights(batch, pi_e, clip)
    per_row = baseline + w * (batch.rewards - q_chosen)
    return OPEResult("dr", float(per_row.mean()), _se(per_row), len(batch))


def evaluate_all(
    batch: LoggedBatch,
    pi_e: np.ndarray,
    q_hat: np.ndarray,
    *,
    clip: float | None = None,
    z: float = 1.96,
    rng: np.random.Generator | None = None,
) -> dict[str, OPEResult]:
    """Run IPS, SNIPS, DM, and DR and return them keyed by estimator name."""
    return {
        "ips": ips(batch, pi_e, clip=clip),
        "snips": snips(batch, pi_e, clip=clip),
        "dm": dm(batch, q_hat, pi_e, rng=rng),
        "dr": dr(batch, q_hat, pi_e, clip=clip),
    }
