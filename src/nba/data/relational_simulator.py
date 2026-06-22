"""A relational door-to-door simulator that *mirrors* the flat one in :mod:`nba.data.simulator`.

This is a drop-in **alternative** ground-truth world, not a rewrite. It emits the exact same
:class:`~nba.schema.BanditEvent` stream (so every learner consumes it unchanged), but its ground
truth carries genuine relational and temporal structure:

- **households** (account nodes grouping nearby doors),
- **neighbor edges** (spatial proximity within ``neighbor_radius_km``),
- **per-prospect interaction histories** (a real, timestamped event log),
- **competitor overlap** (shared-competitor edges among neighboring doors).

The non-relational per-door fields are produced by reusing
:func:`nba.data.simulator.sample_context`, so a *degenerate* world (no edges, no history) reproduces
the flat world exactly.

Oracle isolation (same rule as the flat simulator): :func:`latent_scores`, :func:`true_reward`,
and :func:`true_best_action` are the only ground-truth handles and must never be imported by
``nba.reward``, ``nba.bandits``, ``nba.ope``, ``nba.routing``, ``nba.pipeline``, or ``nba.api``.
The Phase 2 AST guard (``tests/test_ethics.py::test_no_oracle_leak``) scans this module too.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from nba.config import Settings
from nba.data.ames import load_ames
from nba.data.drift import DriftSpec, apply_drift_to_latent
from nba.data.simulator import (
    _BASE_DATE,
    _behavior_scores,
    sample_context,
)
from nba.data.simulator import (
    behavior_policy as _flat_behavior_policy,
)
from nba.data.simulator import (
    frame_to_events as _flat_frame_to_events,
)
from nba.data.simulator import (
    latent_scores as _flat_latent_scores,
)
from nba.data.simulator import (
    logs_to_frame as _flat_logs_to_frame,
)
from nba.routing.territories import cluster_territories
from nba.schema import (
    ACTIONS,
    REWARD,
    Action,
    BanditEvent,
    Outcome,
    ProspectContext,
)

_EARTH_RADIUS_KM = 6371.0088
_OUTCOMES: tuple[Outcome, ...] = tuple(Outcome)
_POSITIVE_OUTCOMES = frozenset({Outcome.APPOINTMENT, Outcome.CLOSED})

# Re-export the flat round-trip so consumers can decode relational logs identically. The extra
# ``household_id`` column (added by :func:`logs_to_frame`) is non-model metadata that
# ``frame_to_events`` ignores, so the data contract is unchanged.
frame_to_events = _flat_frame_to_events


# --------------------------------------------------------------------------------------------- #
# Entity model (the relational substrate)
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Household:
    """An account/household node grouping spatially-coherent doors."""

    household_id: str
    address_ids: list[str]
    centroid: tuple[float, float]


@dataclass(frozen=True)
class Interaction:
    """One historical touch — a temporal edge between a customer and a point in time."""

    address_id: str
    action: Action
    outcome: Outcome
    ts: datetime


@dataclass(frozen=True)
class RelationalWorld:
    """The sampled population plus the relationships that connect it.

    All four effect helpers (:meth:`social_proof`, :meth:`household_momentum`,
    :meth:`history_fatigue`, :meth:`competitor_overlap`) return ``0.0`` when the relevant edge set
    or history is empty, so a world with no structure is numerically identical to the flat world.
    """

    contexts: dict[str, ProspectContext]  # by address_id (insertion order = sample order)
    households: list[Household]
    near_edges: list[tuple[str, str]]  # spatial proximity (<= neighbor_radius_km)
    household_edges: list[tuple[str, str]]  # same-household co-membership
    competitor_edges: list[tuple[str, str]]  # shared-competitor overlap
    histories: dict[str, list[Interaction]]  # per-prospect interaction history

    # Derived adjacency caches (populated in __post_init__; not part of equality/repr).
    _near: dict[str, list[str]] = field(default_factory=dict, compare=False, repr=False)
    _competitor: dict[str, list[str]] = field(default_factory=dict, compare=False, repr=False)
    _addr_household: dict[str, str] = field(default_factory=dict, compare=False, repr=False)
    _household_members: dict[str, list[str]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_near", _adjacency(self.near_edges))
        object.__setattr__(self, "_competitor", _adjacency(self.competitor_edges))
        addr_household: dict[str, str] = {}
        members: dict[str, list[str]] = {}
        for h in self.households:
            members[h.household_id] = list(h.address_ids)
            for addr in h.address_ids:
                addr_household[addr] = h.household_id
        object.__setattr__(self, "_addr_household", addr_household)
        object.__setattr__(self, "_household_members", members)

    # -- relational effect helpers (all zero on an empty/degenerate world) -- #
    def social_proof(self, address_id: str) -> float:
        """Fraction-saturated count of nearby doors with a recent positive interaction."""
        neighbors = self._near.get(address_id, ())
        if not neighbors:
            return 0.0
        positive = sum(1 for nb in neighbors if _has_recent_positive(self.histories.get(nb, ())))
        return float(np.tanh(positive))

    def household_momentum(self, address_id: str) -> float:
        """Saturated count of prior CLOSED touches elsewhere in the same household."""
        hid = self._addr_household.get(address_id)
        if hid is None:
            return 0.0
        closed = 0
        for member in self._household_members.get(hid, ()):
            if member == address_id:
                continue
            closed += sum(
                1 for it in self.histories.get(member, ()) if it.outcome is Outcome.CLOSED
            )
        return float(np.tanh(closed))

    def history_fatigue(self, address_id: str) -> float:
        """Temporal fatigue that decays in over the door's own interaction history length."""
        history = self.histories.get(address_id, ())
        if not history:
            return 0.0
        return float(np.tanh(len(history) / 4.0))

    def competitor_overlap(self, address_id: str) -> float:
        """Saturated count of shared-competitor edges incident to this door."""
        return float(np.tanh(len(self._competitor.get(address_id, ()))))

    def addr_household_map(self) -> dict[str, str]:
        """Return a copy of the ``address_id -> household_id`` map."""
        return dict(self._addr_household)

    def neighbor_fraction(self) -> float:
        """Fraction of doors that have at least one neighbor edge (relational-ness sanity)."""
        if not self.contexts:
            return 0.0
        with_neighbors = sum(1 for addr in self.contexts if self._near.get(addr))
        return with_neighbors / len(self.contexts)


def _adjacency(edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj


def _has_recent_positive(history: list[Interaction] | tuple[Interaction, ...]) -> bool:
    return any(it.outcome in _POSITIVE_OUTCOMES for it in history)


# --------------------------------------------------------------------------------------------- #
# Sampling with structure (vs. flat's i.i.d. draws)
# --------------------------------------------------------------------------------------------- #
def _household_target_size(n: int, settings: Settings) -> int:
    """Doors-per-household target for the spatial clustering."""
    if settings.n_households > 0:
        return max(1, math.ceil(n / settings.n_households))
    return 3  # ~1 household per 3 doors


def _build_near_edges(
    coords: np.ndarray, address_ids: list[str], radius_km: float
) -> list[tuple[str, str]]:
    """Undirected ``near`` edges between doors within ``radius_km`` (great-circle)."""
    if radius_km <= 0.0 or len(address_ids) < 2:
        return []
    from sklearn.neighbors import BallTree

    coords_rad = np.radians(coords)
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = radius_km / _EARTH_RADIUS_KM
    neighbors = tree.query_radius(coords_rad, r=radius_rad)
    edges: set[tuple[str, str]] = set()
    for i, js in enumerate(neighbors):
        for j in js:
            if i < j:
                edges.add((address_ids[i], address_ids[int(j)]))
    return sorted(edges)


def sample_world(n: int, *, settings: Settings, seed: int) -> RelationalWorld:
    """Draw ``n`` Ames-backed doors, group them into households, build near/household/competitor
    edges, and synthesize a per-prospect interaction history.

    Reuses :func:`nba.data.ames.load_ames` and :func:`nba.data.simulator.sample_context` for the
    per-door block so the non-relational fields match the flat world exactly.
    """
    rng = np.random.default_rng(seed)
    ames = load_ames(settings, seed=seed)
    rows = ames.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    contexts_list = [sample_context(rows.iloc[i].to_dict(), rng) for i in range(n)]
    return world_from_contexts(contexts_list, settings=settings, seed=seed)


def world_from_contexts(
    contexts_list: list[ProspectContext], *, settings: Settings, seed: int
) -> RelationalWorld:
    """Build a :class:`RelationalWorld` over *pre-sampled* contexts.

    Edges are derived from the contexts' current ``lat``/``lon``, so callers that reposition doors
    (e.g. the demo's dense-block placement) get a world whose neighbor structure stays consistent
    with the new geography. ``sample_world`` is the convenience wrapper that samples the contexts
    first.
    """
    rng = np.random.default_rng(seed)
    contexts: dict[str, ProspectContext] = {c.address_id: c for c in contexts_list}
    address_ids = [c.address_id for c in contexts_list]
    coords_list = [(c.lat, c.lon) for c in contexts_list]
    n = len(contexts_list)
    coords = np.asarray(coords_list, dtype=np.float64) if coords_list else np.zeros((0, 2))

    # Households: spatially-coherent clusters collapsed to an account node.
    households: list[Household] = []
    household_edges: list[tuple[str, str]] = []
    addr_household: dict[str, str] = {}
    if n > 0:
        target_size = _household_target_size(n, settings)
        territories = cluster_territories(coords_list, target_size=target_size, seed=seed)
        for idx, terr in enumerate(territories):
            members = [address_ids[i] for i in terr.indices]
            hid = f"hh-{idx:05d}"
            households.append(
                Household(household_id=hid, address_ids=members, centroid=terr.centroid)
            )
            for addr in members:
                addr_household[addr] = hid
            for a_i in range(len(members)):
                for b_i in range(a_i + 1, len(members)):
                    household_edges.append((members[a_i], members[b_i]))
    households.sort(key=lambda h: h.household_id)
    household_edges.sort()

    # Neighbor edges: spatial proximity within the configured radius.
    near_edges = _build_near_edges(coords, address_ids, settings.neighbor_radius_km)

    # Competitor overlap: some households sit in a competitor's zone; neighboring doors that are
    # both in a competitor zone share a competitor edge.
    competitor_flag: dict[str, bool] = {}
    for h in households:
        competitor_flag[h.household_id] = bool(rng.random() < settings.competitor_density)
    competitor_edges: list[tuple[str, str]] = []
    if settings.competitor_density > 0.0:
        for a, b in near_edges:
            ha, hb = addr_household.get(a), addr_household.get(b)
            if ha is not None and hb is not None and competitor_flag[ha] and competitor_flag[hb]:
                competitor_edges.append((a, b))
    competitor_edges.sort()

    # History: each prospect gets up to history_len timestamped interactions, consistent with its
    # own ``prior_interactions`` count, drawn from the flat behavior policy + flat ground truth.
    histories: dict[str, list[Interaction]] = {}
    if settings.history_len > 0:
        hist_rng = np.random.default_rng(seed + 1)
        for addr in address_ids:
            ctx = contexts[addr]
            k = min(settings.history_len, int(ctx.prior_interactions))
            if k <= 0:
                continue
            touches: list[Interaction] = []
            for _ in range(k):
                action, _prop = _flat_behavior_policy(ctx, hist_rng)
                outcome = _flat_sample_outcome(ctx, action, hist_rng)
                days_ago = int(hist_rng.integers(1, 180))
                ts = _BASE_DATE - timedelta(days=days_ago)
                touches.append(Interaction(address_id=addr, action=action, outcome=outcome, ts=ts))
            touches.sort(key=lambda it: it.ts)
            histories[addr] = touches

    return RelationalWorld(
        contexts=contexts,
        households=households,
        near_edges=near_edges,
        household_edges=household_edges,
        competitor_edges=competitor_edges,
        histories=histories,
    )


def _flat_sample_outcome(ctx: ProspectContext, action: Action, rng: np.random.Generator) -> Outcome:
    """Draw a flat-world outcome (used only to synthesize histories)."""
    scores = _flat_latent_scores(ctx, action)
    arr = np.array([scores[o] for o in _OUTCOMES], dtype=np.float64)
    arr -= arr.max()
    exp = np.exp(arr)
    probs = exp / exp.sum()
    idx = int(rng.choice(len(_OUTCOMES), p=probs))
    return _OUTCOMES[idx]


# --------------------------------------------------------------------------------------------- #
# Latent ground truth (relational + temporal effects) — ORACLE
# --------------------------------------------------------------------------------------------- #
def latent_scores(
    ctx: ProspectContext,
    action: Action,
    *,
    world: RelationalWorld,
    spec: DriftSpec | None = None,
    event_idx: int = 0,
    n: int = 1,
) -> dict[Outcome, float]:
    """Extend the flat latent scores with effects that REQUIRE the graph.

    - neighbor social proof: nearby recent CLOSED/APPOINTMENT lift APPOINTMENT/CLOSED,
    - household momentum: a prior CLOSED in the same household lifts engagement,
    - interaction fatigue: realistic decay over the door's own history (temporal),
    - competitor overlap: a shared-competitor edge depresses CLOSED.

    Falls back to the flat effects when an edge set is empty, so a degenerate world == flat world.

    When ``spec`` is provided (Phase 18 simulated drift), the drift step is applied
    on top of the relational scores — same wrapper as the flat simulator. ``spec=None``
    leaves the scores unchanged (byte-identical to pre-Phase-18 behaviour).
    """
    scores = _flat_latent_scores(ctx, action)
    if action is Action.SKIP_DOOR:
        return scores  # deterministic null stays null

    addr = ctx.address_id
    proof = world.social_proof(addr)
    momentum = world.household_momentum(addr)
    fatigue = world.history_fatigue(addr)
    competitor = world.competitor_overlap(addr)

    scores[Outcome.APPOINTMENT] += 0.6 * proof + 0.4 * momentum - 0.5 * fatigue - 0.3 * competitor
    scores[Outcome.CLOSED] += 0.5 * proof + 0.3 * momentum - 0.4 * fatigue - 0.6 * competitor
    scores[Outcome.INFO] += 0.2 * momentum
    scores[Outcome.SLAMMED] += 0.6 * fatigue

    if spec is not None:
        scores = apply_drift_to_latent(
            scores, spec=spec, event_idx=event_idx, n=n, ctx=ctx, action=action
        )
    return scores


def outcome_probs(
    ctx: ProspectContext,
    action: Action,
    *,
    world: RelationalWorld,
    spec: DriftSpec | None = None,
    event_idx: int = 0,
    n: int = 1,
) -> dict[Outcome, float]:
    """Softmax of :func:`latent_scores` — the relational ground-truth outcome distribution."""
    scores = latent_scores(ctx, action, world=world, spec=spec, event_idx=event_idx, n=n)
    arr = np.array([scores[o] for o in _OUTCOMES], dtype=np.float64)
    arr -= arr.max()
    exp = np.exp(arr)
    probs = exp / exp.sum()
    return dict(zip(_OUTCOMES, probs.tolist(), strict=True))


def sample_outcome(
    ctx: ProspectContext,
    action: Action,
    rng: np.random.Generator,
    *,
    world: RelationalWorld,
    spec: DriftSpec | None = None,
    event_idx: int = 0,
    n: int = 1,
) -> Outcome:
    """Draw an outcome from the relational ground-truth distribution."""
    probs = outcome_probs(ctx, action, world=world, spec=spec, event_idx=event_idx, n=n)
    idx = int(rng.choice(len(_OUTCOMES), p=np.array([probs[o] for o in _OUTCOMES])))
    return _OUTCOMES[idx]


def true_reward(
    ctx: ProspectContext,
    action: Action,
    *,
    world: RelationalWorld,
    spec: DriftSpec | None = None,
    event_idx: int = 0,
    n: int = 1,
) -> float:
    """Expected reward of ``action`` at ``ctx`` under the relational oracle."""
    probs = outcome_probs(ctx, action, world=world, spec=spec, event_idx=event_idx, n=n)
    return float(sum(probs[o] * REWARD[o] for o in _OUTCOMES))


def true_best_action(
    ctx: ProspectContext,
    *,
    world: RelationalWorld,
    spec: DriftSpec | None = None,
    event_idx: int = 0,
    n: int = 1,
) -> Action:
    """The oracle-optimal action at ``ctx`` (argmax expected reward) under the relational world."""
    return max(
        ACTIONS, key=lambda a: true_reward(ctx, a, world=world, spec=spec, event_idx=event_idx, n=n)
    )


# --------------------------------------------------------------------------------------------- #
# Logging (behavior) policy — identical contract to the flat simulator
# --------------------------------------------------------------------------------------------- #
def action_distribution(
    ctx: ProspectContext, *, world: RelationalWorld, temp: float = 0.5
) -> dict[Action, float]:
    """Full-support softmax logging distribution, mildly nudged by neighbor social proof."""
    scores = _behavior_scores(ctx).astype(np.float64)
    proof = world.social_proof(ctx.address_id)
    if proof > 0.0:
        scores[ACTIONS.index(Action.KNOCK_NOW)] += 0.3 * proof
    scores = scores / temp
    scores -= scores.max()
    exp = np.exp(scores)
    probs = exp / exp.sum()
    return dict(zip(ACTIONS, probs.tolist(), strict=True))


def behavior_policy(
    ctx: ProspectContext,
    rng: np.random.Generator,
    *,
    world: RelationalWorld,
    temp: float = 0.5,
) -> tuple[Action, float]:
    """Sample an action from the logging policy and return ``(action, propensity)`` (``p > 0``)."""
    dist = action_distribution(ctx, world=world, temp=temp)
    probs = np.array([dist[a] for a in ACTIONS], dtype=np.float64)
    idx = int(rng.choice(len(ACTIONS), p=probs))
    return ACTIONS[idx], float(probs[idx])


# --------------------------------------------------------------------------------------------- #
# Event generation + DataFrame I/O (schema-identical to the flat simulator)
# --------------------------------------------------------------------------------------------- #
def generate_logs(
    n: int,
    *,
    settings: Settings,
    seed: int,
    temp: float = 0.5,
    spec: DriftSpec | None = None,
) -> tuple[list[BanditEvent], RelationalWorld]:
    """Generate ``n`` reproducible relational events plus the :class:`RelationalWorld` behind them.

    The emitted :class:`~nba.schema.BanditEvent`s are schema-identical to the flat ones; the extra
    relational structure is carried alongside in the returned world (and graph artifacts).

    When ``spec`` is provided (Phase 18 simulated drift), drift is applied to the relational
    ground-truth outcome distribution. ``spec=None`` keeps the pre-Phase-18 behaviour.
    """
    world = sample_world(n, settings=settings, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    events: list[BanditEvent] = []
    for i, ctx in enumerate(world.contexts.values()):
        clock = _BASE_DATE + timedelta(minutes=int(rng.integers(0, 60 * 24 * 14)))
        action, propensity = behavior_policy(ctx, rng, world=world, temp=temp)
        outcome = sample_outcome(ctx, action, rng, world=world, spec=spec, event_idx=i, n=n)
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
    return events, world


def logs_to_frame(
    events: list[BanditEvent], *, world: RelationalWorld | None = None
) -> pd.DataFrame:
    """Flatten events into the flat parquet schema, plus an optional ``household_id`` column.

    ``household_id`` is non-model metadata: :func:`frame_to_events` ignores it, so the round-trip
    and every downstream learner are unaffected.
    """
    frame = _flat_logs_to_frame(events)
    if world is not None:
        addr_to_hh = world.addr_household_map()
        frame["household_id"] = [addr_to_hh.get(str(a)) for a in frame["ctx.address_id"]]
    return frame
