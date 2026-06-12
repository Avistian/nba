"""Ethics guardrails applied at decision time.

Two protections live here. The first is *structural* and lives elsewhere: models are built only
from :data:`~nba.data.features.ALLOWED_FEATURES`, so no protected attribute or geo/identity field
can reach them. The second is *behavioral* and lives here: in a **sensitive** context we cap how
much the policy is willing to *explore*, because experimenting on someone who has already been
contacted many times risks harassment. Crucially the cap **preserves full support** (every arm
keeps ``p > 0``), so logs stay valid for off-policy evaluation.

This module sees only the (non-protected) context fields; it never touches the simulator oracle.
"""

from __future__ import annotations

import numpy as np

from nba.bandits.base import Policy, sample_from_dist, validate_dist
from nba.config import Settings
from nba.schema import ACTIONS, Action, ProspectContext


def is_sensitive(ctx: ProspectContext, settings: Settings) -> bool:
    """Flag a door where repeated, unsolicited contact warrants caution.

    Defined on a non-protected behavioral signal — the number of prior interactions — not on any
    demographic attribute. A door contacted ``>= settings.sensitive_prior_interactions`` times is
    treated as sensitive.
    """
    return ctx.prior_interactions >= settings.sensitive_prior_interactions


def cap_exploration(dist: dict[Action, float], ceiling: float) -> dict[Action, float]:
    """Shrink a distribution toward its mode so the non-modal ("explore") mass is ``<= ceiling``.

    The modal arm keeps ``1 - ceiling``; the remaining ``ceiling`` is split among the other arms in
    proportion to their original probabilities. Full support is preserved (no arm goes to zero),
    which is what keeps the resulting logs usable by IPS/DR.
    """
    if not 0.0 < ceiling < 1.0:
        raise ValueError("ceiling must be in (0, 1)")

    actions = list(dist.keys())
    probs = np.array([dist[a] for a in actions], dtype=np.float64)
    mode = int(np.argmax(probs))
    explore_mass = 1.0 - probs[mode]

    if explore_mass <= ceiling:
        return dict(dist)  # already conservative enough

    scaled = probs * (ceiling / explore_mass)  # non-modal arms now sum to `ceiling`
    scaled[mode] = 1.0 - ceiling
    return dict(zip(actions, scaled.tolist(), strict=True))


class EthicalPolicy:
    """Wrap any :class:`~nba.bandits.base.Policy` to cap exploration in sensitive contexts.

    In non-sensitive contexts it is a transparent pass-through; in sensitive ones it applies
    :func:`cap_exploration` (when ``settings.cap_exploration_in_sensitive`` is set). Because it only
    reshapes the action distribution, it still satisfies the ``Policy`` protocol.
    """

    def __init__(self, inner: Policy, settings: Settings, *, rng: np.random.Generator) -> None:
        self._inner = inner
        self._settings = settings
        self._rng = rng
        self.name = f"ethical:{inner.name}"

    def action_dist(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> dict[Action, float]:
        dist = self._inner.action_dist(ctx, actions)
        if self._settings.cap_exploration_in_sensitive and is_sensitive(ctx, self._settings):
            dist = cap_exploration(dist, self._settings.sensitive_exploration_ceiling)
        validate_dist(dist)
        return dist

    def recommend(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> tuple[Action, float]:
        return sample_from_dist(self.action_dist(ctx, actions), self._rng)
