"""Domain vocabulary for the NBA prototype.

Defines the action space, the outcome space, the reward map, the prospect context, and the
logged bandit event. Every other module imports its types from here.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Action(StrEnum):
    """A rep's possible move at a door."""

    KNOCK_NOW = "knock_now"
    LEAVE_FLYER = "leave_flyer"
    SKIP_DOOR = "skip_door"
    PITCH_SOLAR = "pitch_solar"
    PITCH_SECURITY = "pitch_security"


#: Canonical action order, frozen for one-hot encoding and ``q_all`` column alignment.
ACTIONS: tuple[Action, ...] = tuple(Action)


class Outcome(StrEnum):
    """The observed result of an action at a door."""

    SLAMMED = "slammed"  # hostile / negative
    NOT_HOME = "not_home"
    INFO = "info_given"  # micro-conversion
    APPOINTMENT = "appointment"  # strong intent
    CLOSED = "closed"  # sale


#: Reward per outcome.
#:
#: ``SLAMMED`` is negative to encode the reputational cost of a bad knock; this makes
#: ``SKIP_DOOR`` a genuine opportunity-cost decision rather than a cost-free option. The map is
#: monotone: SLAMMED < NOT_HOME < INFO < APPOINTMENT < CLOSED.
REWARD: dict[Outcome, float] = {
    Outcome.SLAMMED: -0.2,
    Outcome.NOT_HOME: 0.0,
    Outcome.INFO: 0.1,
    Outcome.APPOINTMENT: 0.3,
    Outcome.CLOSED: 1.0,
}


def reward_for(outcome: Outcome) -> float:
    """Return the reward associated with ``outcome``."""
    return REWARD[outcome]


#: Effort/opportunity prior per action. Used by the router's profit calculation and as a weak
#: heuristic by the logging policy. ``SKIP_DOOR`` is free; knocking and pitching cost the most.
ACTION_COST: dict[Action, float] = {
    Action.SKIP_DOOR: 0.0,
    Action.LEAVE_FLYER: 0.02,
    Action.PITCH_SOLAR: 0.05,
    Action.PITCH_SECURITY: 0.05,
    Action.KNOCK_NOW: 0.05,
}


def action_cost(action: Action) -> float:
    """Return the effort/opportunity cost of ``action``."""
    return ACTION_COST[action]


class ProspectContext(BaseModel):
    """The decision context for a single door, at a single moment.

    Contains no protected attributes (race, religion, etc.); the ethics allow-list in
    :mod:`nba.data.features` provides a second line of defence on what reaches a model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # identity / geo (never enter a model)
    address_id: str
    lat: float
    lon: float

    # prospect block
    property_value: float = Field(ge=0.0)  # USD
    roof_age_years: float = Field(ge=0.0)
    est_income: float = Field(ge=0.0)  # USD/yr
    tenure_years: float = Field(ge=0.0)  # how long resident at address
    prior_interactions: int = Field(ge=0)  # times contacted before

    # environment block
    hour: int = Field(ge=0, le=23)
    dow: int = Field(ge=0, le=6)
    weather: Literal["clear", "rain", "cold", "hot"]
    block_density: float = Field(ge=0.0)  # doors per block
    neighbor_recent_conversion: bool

    # spatial block
    distance_from_rep_km: float = Field(ge=0.0)
    nearby_high_reward_density: float = Field(ge=0.0)  # local density of promising doors


class BanditEvent(BaseModel):
    """A single logged decision, optionally annotated with its observed outcome.

    ``propensity`` is required and strictly positive at construction: this overlap guarantee is
    what makes off-policy evaluation possible. ``reward``/``outcome`` are filled at feedback time.
    """

    model_config = ConfigDict(extra="forbid")

    context: ProspectContext
    action: Action
    propensity: float = Field(gt=0.0, le=1.0)  # p(action | context) under the logging policy
    reward: float | None = None  # filled at feedback time
    outcome: Outcome | None = None
    timestamp: datetime
    decision_id: str  # uuid4; links decision <-> outcome
