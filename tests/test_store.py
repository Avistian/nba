"""Tests for the append-only :class:`EventStore`."""

from __future__ import annotations

import numpy as np
import pytest

from nba.api.store import EventStore, UnknownDecisionError
from nba.config import Settings
from nba.data.simulator import sample_context
from nba.schema import Action, Outcome, ProspectContext, reward_for


def _ctx(seed: int = 1) -> ProspectContext:
    return sample_context(
        {"sale_price": 230_000.0, "year_built": 1995.0}, np.random.default_rng(seed)
    )


def test_append_decision_returns_uuid_and_increments_count(settings: Settings) -> None:
    store = EventStore(settings.db_path)
    assert store.decision_count() == 0

    decision_id = store.append_decision(
        context=_ctx(), action=Action.KNOCK_NOW, propensity=0.42, policy_name="epsilon_greedy"
    )

    assert len(decision_id) == 36  # uuid4 string
    assert store.decision_count() == 1

    events = store.load_events()
    assert len(events) == 1
    assert events[0].propensity == pytest.approx(0.42)
    assert events[0].propensity > 0.0


def test_append_outcome_then_load_yields_reward(settings: Settings) -> None:
    store = EventStore(settings.db_path)
    decision_id = store.append_decision(
        context=_ctx(), action=Action.PITCH_SOLAR, propensity=0.3, policy_name="ucb"
    )

    store.append_outcome(decision_id, Outcome.CLOSED)

    (event,) = store.load_events()
    assert event.outcome is Outcome.CLOSED
    assert event.reward == pytest.approx(reward_for(Outcome.CLOSED))


def test_append_only_latest_outcome_wins(settings: Settings) -> None:
    store = EventStore(settings.db_path)
    decision_id = store.append_decision(
        context=_ctx(), action=Action.KNOCK_NOW, propensity=0.5, policy_name="ucb"
    )

    store.append_outcome(decision_id, Outcome.NOT_HOME)
    store.append_outcome(decision_id, Outcome.CLOSED)  # a correction: a new row, not an update

    assert store.outcome_count() == 2  # both rows retained (append-only)
    (event,) = store.load_events()  # but only one event per decision
    assert event.outcome is Outcome.CLOSED  # latest by id wins
    assert event.reward == pytest.approx(reward_for(Outcome.CLOSED))


def test_decision_without_outcome_loads_unlabeled(settings: Settings) -> None:
    store = EventStore(settings.db_path)
    store.append_decision(
        context=_ctx(), action=Action.SKIP_DOOR, propensity=0.2, policy_name="ucb"
    )

    (event,) = store.load_events()
    assert event.reward is None
    assert event.outcome is None


def test_context_roundtrip_fidelity(settings: Settings) -> None:
    store = EventStore(settings.db_path)
    ctx = _ctx(7)
    store.append_decision(
        context=ctx, action=Action.LEAVE_FLYER, propensity=0.1, policy_name="ucb"
    )

    (event,) = store.load_events()
    assert event.context == ctx


def test_unknown_decision_outcome_raises(settings: Settings) -> None:
    store = EventStore(settings.db_path)
    with pytest.raises(UnknownDecisionError):
        store.append_outcome("does-not-exist", Outcome.INFO)


def test_rejects_nonpositive_propensity(settings: Settings) -> None:
    store = EventStore(settings.db_path)
    with pytest.raises(ValueError):
        store.append_decision(
            context=_ctx(), action=Action.KNOCK_NOW, propensity=0.0, policy_name="ucb"
        )
