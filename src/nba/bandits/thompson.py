"""Thompson sampling via a bootstrap ensemble of reward models.

Thompson sampling needs a *posterior* over ``q(x, a)``. We approximate it cheaply by reusing the
Phase 3 :class:`~nba.reward.model.RewardModel`: fit ``B`` of them, each on an independent
bootstrap resample of the logs (with a per-member seed offset). The spread across members at a
context stands in for posterior uncertainty.

- :meth:`ThompsonSampling.recommend` draws a member uniformly and plays *its* argmax — exactly the
  Thompson "act greedily w.r.t. a posterior sample" step.
- :meth:`ThompsonSampling.action_dist` is the Monte-Carlo estimate of ``P(arm is best)``: the
  fraction of members whose argmax is each arm, floored to full support so OPE keeps overlap.

Fitting ``B`` LightGBMs is the cost, so the ensemble is cacheable to disk like ``RewardModel``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from nba.bandits.base import QEnsemble
from nba.config import Settings
from nba.reward.model import RewardModel
from nba.schema import ACTIONS, Action, BanditEvent, ProspectContext

_META_FILE = "ensemble.json"


class BootstrapEnsemble:
    """``B`` reward models, each fit on a bootstrap resample of the logged events."""

    def __init__(self, members: list[RewardModel]) -> None:
        if not members:
            raise ValueError("ensemble needs at least one member")
        self._members = members

    def __len__(self) -> int:
        return len(self._members)

    @classmethod
    def fit(
        cls,
        events: Sequence[BanditEvent],
        *,
        settings: Settings,
        n_models: int,
        verbose: bool = False,
    ) -> BootstrapEnsemble:
        """Fit ``n_models`` reward models on independent bootstrap resamples of ``events``."""
        if n_models < 1:
            raise ValueError("n_models must be >= 1")
        events = list(events)
        members: list[RewardModel] = []
        for m in range(n_models):
            seed = settings.seed + m
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(events), size=len(events))
            resample = [events[i] for i in idx]
            member_settings = settings.model_copy(update={"seed": seed})
            members.append(RewardModel.fit(resample, settings=member_settings, verbose=verbose))
        return cls(members)

    def q_all_members(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> np.ndarray:
        """Return a ``(B, |actions|)`` matrix of each member's calibrated ``q`` row."""
        return np.vstack([m.q_all(ctx, actions) for m in self._members])

    def save(self, model_dir: Path) -> None:
        """Persist every member under ``model_dir/member_<i>`` plus a small manifest."""
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / _META_FILE).write_text(json.dumps({"n_members": len(self._members)}))
        for i, member in enumerate(self._members):
            member.save(model_dir / f"member_{i}")

    @classmethod
    def load(cls, model_dir: Path) -> BootstrapEnsemble:
        """Load an ensemble previously written by :meth:`save`."""
        meta = json.loads((model_dir / _META_FILE).read_text())
        members = [RewardModel.load(model_dir / f"member_{i}") for i in range(meta["n_members"])]
        return cls(members)


class ThompsonSampling:
    """Thompson sampling over a :class:`BootstrapEnsemble`."""

    name = "thompson"

    def __init__(
        self, ensemble: QEnsemble, *, rng: np.random.Generator, floor: float = 1e-3
    ) -> None:
        if floor < 0.0:
            raise ValueError("floor must be >= 0")
        self._ensemble = ensemble
        self._rng = rng
        self._floor = float(floor)

    def _dist_from_q(
        self, q_members: np.ndarray, actions: tuple[Action, ...]
    ) -> dict[Action, float]:
        """Turn a ``(B, |A|)`` score matrix into a floored ``P(arm is best)`` distribution."""
        argmaxes = np.argmax(q_members, axis=1)
        counts = np.bincount(argmaxes, minlength=len(actions)).astype(np.float64)
        probs = counts / counts.sum() + self._floor
        probs /= probs.sum()
        return dict(zip(actions, probs.tolist(), strict=True))

    def action_dist(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> dict[Action, float]:
        """Return the Monte-Carlo ``P(arm is best)`` distribution, floored to full support."""
        return self._dist_from_q(self._ensemble.q_all_members(ctx, actions), actions)

    def recommend(
        self, ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS
    ) -> tuple[Action, float]:
        """Play the argmax of one uniformly-drawn member; propensity is that arm's ``P(best)``."""
        q_members = self._ensemble.q_all_members(ctx, actions)
        member = int(self._rng.integers(q_members.shape[0]))
        action = actions[int(np.argmax(q_members[member]))]
        dist = self._dist_from_q(q_members, actions)
        return action, dist[action]
