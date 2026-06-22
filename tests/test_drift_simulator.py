"""Tests for the Phase 18 simulated drift module.

Drift is ground-truth (oracle) and must be off by default: ``DriftSpec()`` is a
no-op and the relational simulator with ``spec=None`` matches its pre-Phase-18
output exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nba.config import Settings
from nba.data import relational_simulator as rel
from nba.data import simulator as flat
from nba.data.drift import (
    GRADING_DRIFT_SPEC,
    DriftSpec,
    apply_drift_to_latent,
    generate_logs_for_settings,
    generate_logs_with_drift,
    resolve_drift_spec,
)
from nba.schema import Action, Outcome, ProspectContext


def _ctx(rng: np.random.Generator) -> ProspectContext:
    ames_row = {"sale_price": 230_000.0, "year_built": 1995.0}
    return flat.sample_context(ames_row, rng)


def test_default_spec_is_noop_on_latent() -> None:
    """A default DriftSpec leaves the latent scores unchanged."""
    rng = np.random.default_rng(0)
    ctx = _ctx(rng)
    raw = flat.latent_scores(ctx, Action.KNOCK_NOW)
    drifted = apply_drift_to_latent(
        raw,
        spec=DriftSpec(),
        event_idx=999,
        n=1000,
        ctx=ctx,
        action=Action.KNOCK_NOW,
    )
    assert drifted == pytest.approx(dict(raw))


def test_drift_only_after_at_fraction() -> None:
    """Before ``at_fraction``, even a non-trivial spec is a no-op."""
    rng = np.random.default_rng(1)
    ctx = _ctx(rng)
    raw = flat.latent_scores(ctx, Action.KNOCK_NOW)
    drifted_pre = apply_drift_to_latent(
        raw,
        spec=DriftSpec(at_fraction=0.5, reward_scale=2.0),
        event_idx=10,
        n=1000,  # 10/1000 = 0.01 < 0.5
        ctx=ctx,
        action=Action.KNOCK_NOW,
    )
    assert drifted_pre == pytest.approx(dict(raw))


def test_drift_reward_scale_changes_post_slice() -> None:
    """reward_scale>1 lifts APPOINTMENT/CLOSED mass in the post-drift slice."""
    rng = np.random.default_rng(2)
    ctx = _ctx(rng)
    raw = flat.latent_scores(ctx, Action.KNOCK_NOW)
    drifted = apply_drift_to_latent(
        raw,
        spec=DriftSpec(at_fraction=0.0, reward_scale=2.0),
        event_idx=999,
        n=1000,
        ctx=ctx,
        action=Action.KNOCK_NOW,
    )
    assert drifted[Outcome.APPOINTMENT] == pytest.approx(raw[Outcome.APPOINTMENT] * 2.0)
    assert drifted[Outcome.CLOSED] == pytest.approx(raw[Outcome.CLOSED] * 2.0)


def test_generate_logs_with_default_spec_matches_flat(tmp_path) -> None:
    """generate_logs_with_drift(spec=DriftSpec()) reproduces flat.generate_logs exactly."""
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
    )
    flat_events = flat.generate_logs(500, settings=settings, seed=7)
    drift_events = generate_logs_with_drift(500, settings=settings, seed=7, spec=DriftSpec())
    assert len(flat_events) == len(drift_events)
    for fe, de in zip(flat_events, drift_events, strict=True):
        assert fe.action == de.action
        assert fe.propensity == pytest.approx(de.propensity, rel=1e-12)
        assert fe.reward == pytest.approx(de.reward, rel=1e-12)
        assert fe.outcome == de.outcome


def test_nontrivial_drift_changes_post_slice_mean_reward(tmp_path) -> None:
    """A reward_scale drift bumps mean reward in the post-drift slice, not the pre slice."""
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
    )
    n = 2000
    drift_events = generate_logs_with_drift(
        n,
        settings=settings,
        seed=7,
        spec=DriftSpec(at_fraction=0.5, reward_scale=2.0, knock_evening_boost=0.3),
    )
    flat_events = flat.generate_logs(n, settings=settings, seed=7)
    half = n // 2
    pre_flat = float(np.mean([e.reward for e in flat_events[:half] if e.reward is not None]))
    pre_drift = float(np.mean([e.reward for e in drift_events[:half] if e.reward is not None]))
    post_flat = float(np.mean([e.reward for e in flat_events[half:] if e.reward is not None]))
    post_drift = float(np.mean([e.reward for e in drift_events[half:] if e.reward is not None]))
    # Pre slice must be identical (drift inactive there).
    assert pre_drift == pytest.approx(pre_flat, rel=1e-12)
    # Post slice should differ: lifting APPOINTMENT/CLOSED mass raises expected reward.
    assert post_drift != pytest.approx(post_flat, abs=1e-6)


def test_relational_with_no_spec_unchanged(tmp_path) -> None:
    """The relational simulator with spec=None matches its pre-Phase-18 output exactly."""
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        dataset_mode="relational",
        n_households=20,
        history_len=4,
    )
    events_a, _ = rel.generate_logs(300, settings=settings, seed=11)
    events_b, _ = rel.generate_logs(300, settings=settings, seed=11, spec=None)
    for a, b in zip(events_a, events_b, strict=True):
        assert a.action == b.action
        assert a.reward == pytest.approx(b.reward, rel=1e-12)


def test_use_simulated_drift_flag_wires_grading_spec(tmp_path) -> None:
    """``use_simulated_drift`` enables :data:`GRADING_DRIFT_SPEC` without an explicit ``spec=``."""
    settings_off = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        use_simulated_drift=False,
    )
    settings_on = settings_off.model_copy(update={"use_simulated_drift": True})
    assert resolve_drift_spec(settings_off) is None
    assert resolve_drift_spec(settings_on) == GRADING_DRIFT_SPEC

    flat_off, _ = generate_logs_for_settings(800, settings=settings_off, seed=7)
    flat_on, _ = generate_logs_for_settings(800, settings=settings_on, seed=7)
    mean_off = float(np.mean([e.reward for e in flat_off if e.reward is not None]))
    mean_on = float(np.mean([e.reward for e in flat_on if e.reward is not None]))
    assert mean_on != pytest.approx(mean_off, abs=1e-6)

    rel_settings = settings_on.model_copy(
        update={"dataset_mode": "relational", "n_households": 20, "history_len": 4}
    )
    rel_off, _ = generate_logs_for_settings(
        400,
        settings=settings_off.model_copy(
            update={"dataset_mode": "relational", "n_households": 20, "history_len": 4}
        ),
        seed=11,
    )
    rel_on, _ = generate_logs_for_settings(400, settings=rel_settings, seed=11)
    rel_mean_off = float(np.mean([e.reward for e in rel_off if e.reward is not None]))
    rel_mean_on = float(np.mean([e.reward for e in rel_on if e.reward is not None]))
    assert rel_mean_on != pytest.approx(rel_mean_off, abs=1e-6)


def test_drift_does_not_break_skip_door_determinism() -> None:
    """SKIP_DOOR is a deterministic null; drift must leave it alone."""
    rng = np.random.default_rng(3)
    ctx = _ctx(rng)
    raw = flat.latent_scores(ctx, Action.SKIP_DOOR)
    drifted = apply_drift_to_latent(
        raw,
        spec=DriftSpec(at_fraction=0.0, reward_scale=5.0, knock_evening_boost=1.0),
        event_idx=999,
        n=1000,
        ctx=ctx,
        action=Action.SKIP_DOOR,
    )
    # apply_drift_to_latent returns before the SKIP_DOOR early-return upstream; verify it
    # did not perturb the deterministic null beyond the no-op reward_scale==1 path.
    # We expect SKIP_DOOR scores unchanged because apply_drift_to_latent only touches
    # APPOINTMENT/CLOSED/SLAMMED; the SKIP null mass is on NOT_HOME.
    assert drifted[Outcome.NOT_HOME] == pytest.approx(raw[Outcome.NOT_HOME])
    # Sanity: SKIP_DOOR true_reward is ~0.
    assert math.isclose(flat.true_reward(ctx, Action.SKIP_DOOR), 0.0, abs_tol=1e-9)
