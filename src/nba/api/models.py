"""Pydantic request/response schemas for the HTTP edge.

These are deliberately thin DTOs: domain types (:class:`ProspectContext`, :class:`Action`,
:class:`Outcome`) are reused directly so validation and the action/outcome vocabulary stay in one
place (:mod:`nba.schema`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nba.schema import Action, Outcome, ProspectContext


class RecommendRequest(BaseModel):
    """Ask for an action at a single door."""

    context: ProspectContext


class RecommendResponse(BaseModel):
    """The logged decision: its id, the chosen action, its propensity, and the raw q-values."""

    decision_id: str
    action: Action
    propensity: float = Field(gt=0.0, le=1.0)
    q_values: dict[Action, float]


class FeedbackRequest(BaseModel):
    """Report the observed outcome for a previously recommended decision."""

    decision_id: str
    outcome: Outcome


class RouteRequest(BaseModel):
    """Plan a walkable route over a set of candidate doors."""

    contexts: list[ProspectContext]


class RouteStop(BaseModel):
    """A serviced door and its position in the visiting sequence."""

    address_id: str
    lat: float
    lon: float
    order: int


class RouteResponse(BaseModel):
    """An ordered, walkable plan plus the doors dropped as not worth the walk."""

    stops: list[RouteStop]
    dropped: list[str]
    total_time_s: float
    total_profit: float


class HealthResponse(BaseModel):
    """Liveness plus the active policy and how many decisions have been logged."""

    status: Literal["ok"]
    policy: str
    decisions: int
