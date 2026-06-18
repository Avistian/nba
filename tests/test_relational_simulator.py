"""Tests for the relational simulator: mirror contract, positivity, reproducibility, real
relational signal, and the degenerate==flat guarantee."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nba.config import Settings
from nba.data import relational_simulator as rel
from nba.data import simulator as flat
from nba.data.relational_simulator import Interaction, RelationalWorld
from nba.schema import ACTIONS, Action, BanditEvent, Outcome


def _ctx(seed: int = 1, **overrides: object):
    base = flat.sample_context(
        {"sale_price": 230_000.0, "year_built": 1995.0}, np.random.default_rng(seed)
    )
    return base.model_copy(update=overrides) if overrides else base


# --------------------------------------------------------------------------------------------- #
# Mirror contract + positivity
# --------------------------------------------------------------------------------------------- #
def test_emitted_events_match_schema_and_round_trip() -> None:
    settings = Settings()
    events, world = rel.generate_logs(400, settings=settings, seed=7)

    assert all(isinstance(e, BanditEvent) for e in events)
    frame = rel.logs_to_frame(events, world=world)
    assert "household_id" in frame.columns  # additive, non-model metadata

    # The relational logs decode through the flat round-trip identically.
    decoded = rel.frame_to_events(frame)
    assert len(decoded) == len(events)
    for a, b in zip(events, decoded, strict=True):
        assert a.context == b.context
        assert a.action == b.action
        assert a.outcome == b.outcome
        assert a.propensity == pytest.approx(b.propensity)


def test_positivity_and_full_arm_support() -> None:
    settings = Settings()
    events, _world = rel.generate_logs(600, settings=settings, seed=7)
    assert all(e.propensity > 0.0 for e in events)
    seen = {e.action for e in events}
    assert seen == set(ACTIONS)


def test_reproducible_frames_and_world() -> None:
    settings = Settings()
    events_a, world_a = rel.generate_logs(300, settings=settings, seed=7)
    events_b, world_b = rel.generate_logs(300, settings=settings, seed=7)

    frame_a = rel.logs_to_frame(events_a, world=world_a)
    frame_b = rel.logs_to_frame(events_b, world=world_b)
    pd.testing.assert_frame_equal(frame_a, frame_b)
    assert world_a == world_b


def test_world_is_actually_relational() -> None:
    settings = Settings()
    world = rel.sample_world(1000, settings=settings, seed=7)
    assert world.households, "expected household nodes"
    assert world.near_edges, "expected a non-trivial neighbor edge set"
    assert world.neighbor_fraction() > 0.0


# --------------------------------------------------------------------------------------------- #
# Real relational signal (oracle-only assertions)
# --------------------------------------------------------------------------------------------- #
def _two_door_world(*, near: bool, neighbor_closed: bool, competitor: bool) -> RelationalWorld:
    door = _ctx(seed=1)
    neighbor = _ctx(seed=2)
    contexts = {door.address_id: door, neighbor.address_id: neighbor}
    lo, hi = sorted((door.address_id, neighbor.address_id))
    pair: tuple[str, str] = (lo, hi)
    near_edges: list[tuple[str, str]] = [pair] if near else []
    competitor_edges: list[tuple[str, str]] = [pair] if competitor else []
    histories: dict[str, list[Interaction]] = {}
    if neighbor_closed:
        histories[neighbor.address_id] = [
            Interaction(neighbor.address_id, Action.KNOCK_NOW, Outcome.CLOSED, datetime(2026, 1, 1))
        ]
    return RelationalWorld(
        contexts=contexts,
        households=[],
        near_edges=near_edges,
        household_edges=[],
        competitor_edges=competitor_edges,
        histories=histories,
    )


def test_neighbor_social_proof_lifts_reward() -> None:
    no_proof = _two_door_world(near=True, neighbor_closed=False, competitor=False)
    with_proof = _two_door_world(near=True, neighbor_closed=True, competitor=False)
    door_id = next(iter(no_proof.contexts))
    door = no_proof.contexts[door_id]

    base = rel.true_reward(door, Action.KNOCK_NOW, world=no_proof)
    lifted = rel.true_reward(no_proof.contexts[door_id], Action.KNOCK_NOW, world=with_proof)
    assert lifted > base


def test_competitor_overlap_lowers_reward() -> None:
    plain = _two_door_world(near=True, neighbor_closed=False, competitor=False)
    overlapped = _two_door_world(near=True, neighbor_closed=False, competitor=True)
    door_id = next(iter(plain.contexts))
    door = plain.contexts[door_id]

    base = rel.true_reward(door, Action.KNOCK_NOW, world=plain)
    depressed = rel.true_reward(door, Action.KNOCK_NOW, world=overlapped)
    assert depressed < base


# --------------------------------------------------------------------------------------------- #
# Degenerate world == flat world
# --------------------------------------------------------------------------------------------- #
def test_degenerate_world_matches_flat_oracle() -> None:
    settings = Settings(neighbor_radius_km=0.0, competitor_density=0.0, history_len=0)
    world = rel.sample_world(200, settings=settings, seed=7)
    assert not world.near_edges
    assert not world.competitor_edges
    assert not world.histories

    for ctx in list(world.contexts.values())[:50]:
        for action in ACTIONS:
            assert rel.true_reward(ctx, action, world=world) == pytest.approx(
                flat.true_reward(ctx, action)
            )
