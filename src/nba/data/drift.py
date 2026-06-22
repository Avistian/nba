"""Ground-truth drift injection for demos and grading only (ORACLE module).

This module mirrors the oracle-isolation rule of :mod:`nba.data.simulator` and
:mod:`nba.data.relational_simulator`: it must NEVER be imported from
``nba.reward``, ``nba.bandits``, ``nba.ope``, ``nba.routing``, ``nba.pipeline``,
``nba.api``, or ``nba.monitoring``. The Phase 2 AST guard
(``tests/test_ethics.py::test_no_oracle_leak``) scans this module too.

A :class:`DriftSpec` injects a step change partway through log generation. The
default spec (``DriftSpec()``) is a no-op: ``generate_logs_with_drift`` with the
default spec reproduces :func:`nba.data.simulator.generate_logs` byte-for-byte
on the same seed.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from nba.data.ames import load_ames
from nba.data.simulator import _BASE_DATE, sample_context
from nba.data.simulator import behavior_policy as _flat_behavior_policy
from nba.data.simulator import latent_scores as _flat_latent_scores
from nba.schema import REWARD, Action, BanditEvent, Outcome, ProspectContext

_OUTCOMES: tuple[Outcome, ...] = tuple(Outcome)


@dataclass(frozen=True)
class DriftSpec:
    """Inject a step change partway through log generation.

    All multipliers apply only to events with ``event_idx/n >= at_fraction``.
    The defaults (``1.0``/``0.0``) leave the latent scores unchanged, so a
    default spec reproduces the flat simulator exactly.
    """

    at_fraction: float = 0.5  # fraction of events before drift kicks in
    reward_scale: float = 1.0  # multiply appointment/closed mass post-drift
    knock_evening_boost: float = 0.0  # add to KNOCK_NOW evening effect post-drift
    weather_slam_mult: float = 1.0  # multiply bad-weather slam mass post-drift


def _is_active(spec: DriftSpec, event_idx: int, n: int) -> bool:
    """True when ``event_idx/n >= spec.at_fraction``."""
    if n <= 0:
        return False
    return (event_idx / n) >= spec.at_fraction


def apply_drift_to_latent(
    scores: Mapping[Outcome, float],
    *,
    spec: DriftSpec,
    event_idx: int,
    n: int,
    ctx: ProspectContext,
    action: Action,
) -> dict[Outcome, float]:
    """Post-process a latent-scores dict with the spec's drift step.

    A no-op when ``DriftSpec()`` is at defaults or the event is before the
    ``at_fraction`` cut. Returns a fresh dict; never mutates the input.
    """
    out = dict(scores)
    if not _is_active(spec, event_idx, n):
        return out

    # No drift at all -> return without further work (keeps default spec byte-identical).
    if (
        spec.reward_scale == 1.0
        and spec.knock_evening_boost == 0.0
        and spec.weather_slam_mult == 1.0
    ):
        return out

    if spec.reward_scale != 1.0:
        out[Outcome.APPOINTMENT] = out[Outcome.APPOINTMENT] * spec.reward_scale
        out[Outcome.CLOSED] = out[Outcome.CLOSED] * spec.reward_scale

    if spec.knock_evening_boost != 0.0 and action is Action.KNOCK_NOW:
        evening = 17 <= ctx.hour <= 19
        if evening:
            out[Outcome.APPOINTMENT] = out[Outcome.APPOINTMENT] + spec.knock_evening_boost

    if spec.weather_slam_mult != 1.0:
        bad_weather = ctx.weather in ("rain", "cold")
        if bad_weather:
            out[Outcome.SLAMMED] = out[Outcome.SLAMMED] * spec.weather_slam_mult

    return out


def _outcome_probs(
    ctx: ProspectContext, action: Action, *, spec: DriftSpec, event_idx: int, n: int
) -> dict[Outcome, float]:
    """Softmax of the drift-perturbed latent scores — the ground-truth distribution."""
    raw = _flat_latent_scores(ctx, action)
    drifted = apply_drift_to_latent(
        raw, spec=spec, event_idx=event_idx, n=n, ctx=ctx, action=action
    )
    arr = np.array([drifted[o] for o in _OUTCOMES], dtype=np.float64)
    arr -= arr.max()
    exp = np.exp(arr)
    probs = exp / exp.sum()
    return dict(zip(_OUTCOMES, probs.tolist(), strict=True))


def _sample_outcome(
    ctx: ProspectContext,
    action: Action,
    rng: np.random.Generator,
    *,
    spec: DriftSpec,
    event_idx: int,
    n: int,
) -> Outcome:
    """Draw an outcome from the drift-perturbed ground-truth distribution."""
    probs = _outcome_probs(ctx, action, spec=spec, event_idx=event_idx, n=n)
    idx = int(rng.choice(len(_OUTCOMES), p=np.array([probs[o] for o in _OUTCOMES])))
    return _OUTCOMES[idx]


def generate_logs_with_drift(
    n: int,
    *,
    settings: Settings,  # type: ignore[name-defined]  # noqa: F821
    seed: int,
    spec: DriftSpec,
    temp: float = 0.5,
) -> list[BanditEvent]:
    """Generate ``n`` reproducible logged events with drift applied per :class:`DriftSpec`.

    Mirrors :func:`nba.data.simulator.generate_logs` exactly when ``spec == DriftSpec()``.
    The logging policy is the flat behavior policy (unchanged by drift); only the
    ground-truth outcome distribution is perturbed.
    """
    # Local import keeps the import graph clean of cycles when ``Settings`` evolves.
    from nba.config import Settings  # noqa: PLC0415

    assert isinstance(settings, Settings)

    rng = np.random.default_rng(seed)
    ames = load_ames(settings, seed=seed)
    rows = ames.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    events: list[BanditEvent] = []
    for i in range(n):
        ctx = sample_context(rows.iloc[i].to_dict(), rng)
        clock = _BASE_DATE + timedelta(minutes=int(rng.integers(0, 60 * 24 * 14)))
        action, propensity = _flat_behavior_policy(ctx, rng, temp=temp)
        outcome = _sample_outcome(ctx, action, rng, spec=spec, event_idx=i, n=n)
        events.append(
            BanditEvent(
                context=ctx,
                action=action,
                propensity=propensity,
                reward=REWARD[outcome],
                outcome=outcome,
                timestamp=clock,
                decision_id=str(uuid.UUID(int=int(rng.integers(0, 2**63)))),
            )
        )
    return events
