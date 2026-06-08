"""Tests for :mod:`nba.schema`."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from nba.schema import (
    ACTIONS,
    REWARD,
    Action,
    BanditEvent,
    Outcome,
    ProspectContext,
    action_cost,
    reward_for,
)


def _context(**overrides: object) -> ProspectContext:
    base: dict[str, object] = {
        "address_id": "a-1",
        "lat": 42.0,
        "lon": -93.0,
        "property_value": 250_000.0,
        "roof_age_years": 12.0,
        "est_income": 85_000.0,
        "tenure_years": 4.0,
        "prior_interactions": 1,
        "hour": 18,
        "dow": 2,
        "weather": "clear",
        "block_density": 20.0,
        "neighbor_recent_conversion": True,
        "distance_from_rep_km": 0.4,
        "nearby_high_reward_density": 0.7,
    }
    base.update(overrides)
    return ProspectContext(**base)  # type: ignore[arg-type]


def test_actions_order_and_length() -> None:
    assert tuple(Action) == ACTIONS
    assert len(ACTIONS) == 5


def test_reward_map_strictly_ordered() -> None:
    ordered = [
        REWARD[Outcome.SLAMMED],
        REWARD[Outcome.NOT_HOME],
        REWARD[Outcome.INFO],
        REWARD[Outcome.APPOINTMENT],
        REWARD[Outcome.CLOSED],
    ]
    assert ordered == sorted(ordered)
    assert len(set(ordered)) == len(ordered)
    assert reward_for(Outcome.CLOSED) == pytest.approx(1.0)


def test_action_cost_ordering() -> None:
    assert action_cost(Action.SKIP_DOOR) == 0.0
    assert action_cost(Action.KNOCK_NOW) > action_cost(Action.LEAVE_FLYER)


@pytest.mark.parametrize("bad_propensity", [0.0, 1.5, -0.1])
def test_bandit_event_rejects_invalid_propensity(bad_propensity: float) -> None:
    with pytest.raises(ValidationError):
        BanditEvent(
            context=_context(),
            action=Action.KNOCK_NOW,
            propensity=bad_propensity,
            timestamp=datetime(2026, 6, 8, 18, 0, 0),
            decision_id="d-1",
        )


def test_bandit_event_valid_minimal() -> None:
    event = BanditEvent(
        context=_context(),
        action=Action.KNOCK_NOW,
        propensity=0.4,
        timestamp=datetime(2026, 6, 8, 18, 0, 0),
        decision_id="d-1",
    )
    assert event.reward is None
    assert event.outcome is None
    assert 0.0 < event.propensity <= 1.0
    assert event.decision_id == "d-1"


def test_context_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        _context(unexpected_field=1)


def test_context_rejects_out_of_range_hour() -> None:
    with pytest.raises(ValidationError):
        _context(hour=24)
