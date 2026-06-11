"""Promotion gate: ship a candidate policy only if OPE says it beats the logging baseline.

The gate is the safety valve before "the bandit proposes" reaches the field. It estimates the
candidate's value off-policy (primary = **DR**, with IPS/DM reported for transparency) and
promotes only when the candidate's *lower confidence bound* clears the baseline by a margin — a
deliberately conservative rule, since acting on an over-optimistic estimate is the expensive
failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nba.bandits.base import Policy
from nba.ope.estimators import LoggedBatch, OPEResult, eval_action_matrix, evaluate_all


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict plus the evidence behind it."""

    promote: bool
    candidate: dict[str, OPEResult]  # estimator name → result for the candidate policy
    baseline_value: float
    lift: float  # candidate DR value − baseline_value
    lower_bound: float  # candidate DR value − z·se
    reason: str


class PromotionGate:
    """Decide whether to promote a candidate policy over the logging baseline."""

    def __init__(self, *, z: float, min_lift: float) -> None:
        self._z = float(z)
        self._min_lift = float(min_lift)

    def evaluate(
        self,
        candidate: Policy,
        batch: LoggedBatch,
        q_hat: np.ndarray,
        *,
        baseline_value: float,
        clip: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> GateDecision:
        """Estimate ``candidate`` off-policy and promote iff its DR lower bound beats baseline."""
        pi_e = eval_action_matrix(candidate, batch.contexts)
        results = evaluate_all(batch, pi_e, q_hat, clip=clip, z=self._z, rng=rng)

        dr_res = results["dr"]
        lower_bound = dr_res.value - self._z * dr_res.std_err
        lift = dr_res.value - baseline_value
        threshold = baseline_value + self._min_lift
        promote = bool(lower_bound > threshold)

        verdict = "PROMOTE" if promote else "HOLD"
        reason = (
            f"{verdict}: DR={dr_res.value:.4f} (lb={lower_bound:.4f}, "
            f"lift={lift:+.4f}) vs baseline+min_lift={threshold:.4f}"
        )

        # Disagreement between the unbiased-but-noisy IPS and the low-variance-but-biased DM is a
        # signal that one of the assumptions (overlap / q̂ accuracy) is shaky — surface it.
        ips_v, dm_v = results["ips"].value, results["dm"].value
        scale = max(abs(baseline_value), 1e-6)
        if abs(ips_v - dm_v) > 0.5 * scale:
            reason += f" | caution: IPS({ips_v:.4f}) and DM({dm_v:.4f}) disagree"

        return GateDecision(
            promote=promote,
            candidate=results,
            baseline_value=float(baseline_value),
            lift=float(lift),
            lower_bound=float(lower_bound),
            reason=reason,
        )
