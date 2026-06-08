"""A self-contained door-to-door simulator: the ground-truth world for the prototype.

It (a) draws realistic :class:`~nba.schema.ProspectContext`s from Ames housing rows plus sampled
environment, (b) defines a latent conversion model and the ``true_reward`` oracle, and (c) runs a
stochastic logging policy that **records propensity** so the emitted logs support off-policy
evaluation.

Oracle isolation: :func:`latent_scores`, :func:`true_reward`, and :func:`true_best_action` are the
only ground-truth handles. They must never be imported by ``nba.reward``, ``nba.bandits``, or
``nba.ope`` — those modules only ever see logged ``(context, action, reward, propensity)`` tuples.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from nba.config import Settings
from nba.data.ames import load_ames
from nba.schema import (
    ACTIONS,
    REWARD,
    Action,
    BanditEvent,
    Outcome,
    ProspectContext,
    action_cost,
)

Weather = Literal["clear", "rain", "cold", "hot"]
_WEATHER_LEVELS: tuple[Weather, ...] = ("clear", "rain", "cold", "hot")
_OUTCOMES: tuple[Outcome, ...] = tuple(Outcome)
_BASE_DATE = datetime(2026, 6, 1, 8, 0, 0)


class Environment(TypedDict):
    """The sampled time/weather/neighborhood block of a context."""

    hour: int
    dow: int
    weather: Weather
    block_density: float
    neighbor_recent_conversion: bool


# --------------------------------------------------------------------------------------------- #
# Environment + context sampling
# --------------------------------------------------------------------------------------------- #
def sample_environment(rng: np.random.Generator) -> Environment:
    """Sample the time/weather/neighborhood block of a context."""
    weather_idx = int(rng.choice(len(_WEATHER_LEVELS), p=[0.55, 0.2, 0.15, 0.1]))
    return Environment(
        hour=int(rng.integers(8, 21)),  # canvassing window 08:00–20:00
        dow=int(rng.integers(0, 7)),
        weather=_WEATHER_LEVELS[weather_idx],
        block_density=float(rng.uniform(5.0, 40.0)),
        neighbor_recent_conversion=bool(rng.random() < 0.25),
    )


def sample_context(ames_row: Mapping[str, float], rng: np.random.Generator) -> ProspectContext:
    """Join an Ames property row with sampled environment + spatial features into a context."""
    env = sample_environment(rng)
    sale_price = float(ames_row["sale_price"])
    year_built = float(ames_row.get("year_built", 1980.0) or 1980.0)
    roof_age = float(np.clip(2026.0 - year_built, 0.0, 120.0))
    # income loosely tracks property value, with noise.
    est_income = float(np.clip(sale_price * 0.28 + rng.normal(0.0, 12_000.0), 15_000.0, 500_000.0))
    lat = float(42.03 + rng.normal(0.0, 0.03))
    lon = float(-93.62 + rng.normal(0.0, 0.03))
    return ProspectContext(
        address_id=str(uuid.UUID(int=int(rng.integers(0, 2**63)))),
        lat=lat,
        lon=lon,
        property_value=sale_price,
        roof_age_years=roof_age,
        est_income=est_income,
        tenure_years=float(np.clip(rng.exponential(6.0), 0.0, 60.0)),
        prior_interactions=int(rng.poisson(0.6)),
        hour=env["hour"],
        dow=env["dow"],
        weather=env["weather"],
        block_density=env["block_density"],
        neighbor_recent_conversion=env["neighbor_recent_conversion"],
        distance_from_rep_km=float(np.clip(rng.exponential(0.5), 0.0, 10.0)),
        nearby_high_reward_density=float(rng.uniform(0.0, 1.0)),
    )


# --------------------------------------------------------------------------------------------- #
# Latent ground truth (oracle) — DO NOT import from reward/bandits/ope
# --------------------------------------------------------------------------------------------- #
def latent_scores(ctx: ProspectContext, action: Action) -> dict[Outcome, float]:
    """Unnormalized per-outcome logits encoding documented interaction effects.

    Effects:
      - ``KNOCK_NOW`` in the evening (17–19h) lifts APPOINTMENT/CLOSED.
      - Long ``tenure_years`` lifts engagement; high ``property_value`` lifts solar fit.
      - ``neighbor_recent_conversion`` adds social proof.
      - ``PITCH_SECURITY`` does better with low ``block_density`` / high income.
      - Bad weather lifts SLAMMED/NOT_HOME.
      - Many ``prior_interactions`` => diminishing returns and more SLAMMED.
      - ``SKIP_DOOR`` is a deterministic null (all mass on NOT_HOME-like).
    """
    s: dict[Outcome, float] = {o: 0.0 for o in _OUTCOMES}

    if action is Action.SKIP_DOOR:
        # Deterministic null: skipping produces no engagement (reward 0).
        s[Outcome.NOT_HOME] = 12.0
        for o in _OUTCOMES:
            if o is not Outcome.NOT_HOME:
                s[o] = -12.0
        return s

    evening = 17 <= ctx.hour <= 19
    tenure = np.tanh(ctx.tenure_years / 8.0)
    wealth = np.tanh((ctx.property_value - 180_000.0) / 150_000.0)
    income = np.tanh((ctx.est_income - 60_000.0) / 60_000.0)
    fatigue = np.tanh(ctx.prior_interactions / 3.0)
    bad_weather = ctx.weather in ("rain", "cold")
    social = 1.0 if ctx.neighbor_recent_conversion else 0.0

    # Base engagement: more positive when answered, baseline absence.
    s[Outcome.NOT_HOME] = 0.6 - 0.5 * tenure + (0.5 if bad_weather else 0.0)
    s[Outcome.SLAMMED] = -0.2 + 0.8 * fatigue + (0.4 if bad_weather else 0.0)
    s[Outcome.INFO] = 0.2 + 0.4 * tenure + 0.3 * social
    s[Outcome.APPOINTMENT] = -0.4 + 0.6 * tenure + 0.5 * social - 0.6 * fatigue
    s[Outcome.CLOSED] = -1.2 + 0.5 * tenure + 0.4 * social - 0.5 * fatigue

    if action is Action.KNOCK_NOW:
        if evening:
            s[Outcome.APPOINTMENT] += 0.8
            s[Outcome.CLOSED] += 0.5
        s[Outcome.INFO] += 0.3
        s[Outcome.SLAMMED] += 0.2  # knocking risks a slam
    elif action is Action.LEAVE_FLYER:
        # Low-risk, low-reward: rarely a slam, rarely a close.
        s[Outcome.SLAMMED] -= 0.6
        s[Outcome.INFO] += 0.1
        s[Outcome.APPOINTMENT] -= 0.3
        s[Outcome.CLOSED] -= 0.6
    elif action is Action.PITCH_SOLAR:
        s[Outcome.APPOINTMENT] += 0.4 + 0.8 * wealth
        s[Outcome.CLOSED] += 0.2 + 0.7 * wealth
    elif action is Action.PITCH_SECURITY:
        low_density = np.tanh((20.0 - ctx.block_density) / 15.0)
        s[Outcome.APPOINTMENT] += 0.3 + 0.5 * income + 0.4 * low_density
        s[Outcome.CLOSED] += 0.1 + 0.4 * income + 0.3 * low_density

    return s


def outcome_probs(ctx: ProspectContext, action: Action) -> dict[Outcome, float]:
    """Softmax of :func:`latent_scores` — the ground-truth outcome distribution."""
    scores = latent_scores(ctx, action)
    arr = np.array([scores[o] for o in _OUTCOMES], dtype=np.float64)
    arr -= arr.max()
    exp = np.exp(arr)
    probs = exp / exp.sum()
    return dict(zip(_OUTCOMES, probs.tolist(), strict=True))


def sample_outcome(ctx: ProspectContext, action: Action, rng: np.random.Generator) -> Outcome:
    """Draw an outcome from the ground-truth distribution."""
    probs = outcome_probs(ctx, action)
    idx = int(rng.choice(len(_OUTCOMES), p=np.array([probs[o] for o in _OUTCOMES])))
    return _OUTCOMES[idx]


def true_reward(ctx: ProspectContext, action: Action) -> float:
    """Expected reward of ``action`` at ``ctx`` under the oracle (used for regret/OPE truth)."""
    probs = outcome_probs(ctx, action)
    return float(sum(probs[o] * REWARD[o] for o in _OUTCOMES))


def true_best_action(ctx: ProspectContext) -> Action:
    """The oracle-optimal action at ``ctx`` (argmax expected reward)."""
    return max(ACTIONS, key=lambda a: true_reward(ctx, a))


# --------------------------------------------------------------------------------------------- #
# Logging (behavior) policy — records propensity, full support
# --------------------------------------------------------------------------------------------- #
def _behavior_scores(ctx: ProspectContext) -> np.ndarray:
    """Cheap heuristic score per action used by the logging policy (NOT the oracle)."""
    evening = 17 <= ctx.hour <= 19
    scores = np.zeros(len(ACTIONS), dtype=np.float64)
    for i, a in enumerate(ACTIONS):
        score = -action_cost(a)
        if a is Action.KNOCK_NOW and evening:
            score += 0.6
        if a is Action.PITCH_SOLAR and ctx.property_value > 220_000.0:
            score += 0.4
        if a is Action.LEAVE_FLYER and ctx.weather in ("rain", "cold"):
            score += 0.3
        if a is Action.SKIP_DOOR and ctx.prior_interactions >= 3:
            score += 0.5
        scores[i] = score
    return scores


def action_distribution(ctx: ProspectContext, *, temp: float = 0.5) -> dict[Action, float]:
    """Full-support softmax logging distribution (every arm gets ``p > 0``)."""
    scores = _behavior_scores(ctx) / temp
    scores -= scores.max()
    exp = np.exp(scores)
    probs = exp / exp.sum()
    return dict(zip(ACTIONS, probs.tolist(), strict=True))


def behavior_policy(
    ctx: ProspectContext, rng: np.random.Generator, *, temp: float = 0.5
) -> tuple[Action, float]:
    """Sample an action from the logging policy and return ``(action, propensity)``."""
    dist = action_distribution(ctx, temp=temp)
    probs = np.array([dist[a] for a in ACTIONS], dtype=np.float64)
    idx = int(rng.choice(len(ACTIONS), p=probs))
    action = ACTIONS[idx]
    return action, float(probs[idx])


# --------------------------------------------------------------------------------------------- #
# Event generation
# --------------------------------------------------------------------------------------------- #
def simulate_event(
    ctx: ProspectContext, rng: np.random.Generator, clock: datetime, *, temp: float = 0.5
) -> BanditEvent:
    """Produce one fully-labeled logged event for ``ctx``."""
    action, propensity = behavior_policy(ctx, rng, temp=temp)
    outcome = sample_outcome(ctx, action, rng)
    return BanditEvent(
        context=ctx,
        action=action,
        propensity=propensity,
        reward=REWARD[outcome],
        outcome=outcome,
        timestamp=clock,
        decision_id=str(uuid.UUID(int=int(rng.integers(0, 2**63)))),
    )


def generate_logs(n: int, *, settings: Settings, seed: int, temp: float = 0.5) -> list[BanditEvent]:
    """Generate ``n`` reproducible logged events from sampled Ames-backed contexts."""
    rng = np.random.default_rng(seed)
    ames = load_ames(settings, seed=seed)
    rows = ames.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    events: list[BanditEvent] = []
    for i in range(n):
        ctx = sample_context(rows.iloc[i].to_dict(), rng)
        clock = _BASE_DATE + timedelta(minutes=int(rng.integers(0, 60 * 24 * 14)))
        events.append(simulate_event(ctx, rng, clock, temp=temp))
    return events


def logs_to_frame(events: list[BanditEvent]) -> pd.DataFrame:
    """Flatten events into a parquet-friendly frame with ``ctx.*`` columns."""
    records: list[dict[str, object]] = []
    for e in events:
        rec: dict[str, object] = {f"ctx.{k}": v for k, v in e.context.model_dump().items()}
        rec.update(
            {
                "action": e.action.value,
                "propensity": e.propensity,
                "reward": e.reward,
                "outcome": e.outcome.value if e.outcome is not None else None,
                "decision_id": e.decision_id,
                "timestamp": e.timestamp,
                "lat": e.context.lat,
                "lon": e.context.lon,
            }
        )
        records.append(rec)
    return pd.DataFrame.from_records(records)


def frame_to_events(frame: pd.DataFrame) -> list[BanditEvent]:
    """Reconstruct :class:`~nba.schema.BanditEvent`s from a flattened log frame."""
    ctx_cols = [c for c in frame.columns if c.startswith("ctx.")]
    events: list[BanditEvent] = []
    for record in frame.to_dict(orient="records"):
        ctx_kwargs = {c[len("ctx.") :]: record[c] for c in ctx_cols}
        context = ProspectContext(**ctx_kwargs)
        outcome = record.get("outcome")
        reward = record.get("reward")
        events.append(
            BanditEvent(
                context=context,
                action=Action(record["action"]),
                propensity=float(record["propensity"]),
                reward=None if reward is None or pd.isna(reward) else float(reward),
                outcome=None if outcome is None or pd.isna(outcome) else Outcome(outcome),
                timestamp=record["timestamp"],
                decision_id=str(record["decision_id"]),
            )
        )
    return events
