"""Tests for :mod:`nba.reward.decision_focused` (Phase 12).

Covers the two safety rails (default-off is byte-identical; SPO+ is still a calibrated ``QModel``),
the two on-ramps' mechanics (reweighting shifts attention; SPO+'s zero-step identity), and the
headline claim: decision-focused training lowers **decision regret** (route value vs an oracle that
knew the true prizes) without breaking calibration. The oracle is used here for *grading only*, as
in the demo and other tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from nba.config import Settings
from nba.data.ames import load_ames
from nba.data.features import featurize
from nba.data.simulator import generate_logs, sample_context, true_reward
from nba.reward.decision_focused import decision_aware_weights
from nba.reward.model import RewardModel
from nba.schema import ACTIONS, ProspectContext


def _probe_contexts(n: int) -> list[ProspectContext]:
    rng = np.random.default_rng(123)
    return [sample_context({"sale_price": 230_000.0, "year_built": 1995.0}, rng) for _ in range(n)]


# --------------------------------------------------------------------------------------------- #
# Safety rail 1: default off is a byte-exact no-op
# --------------------------------------------------------------------------------------------- #
def test_default_off_is_byte_identical() -> None:
    settings = Settings(seed=7)
    events = generate_logs(800, settings=settings, seed=7)
    probe = _probe_contexts(30)

    a = RewardModel.fit(events, settings=settings)
    b = RewardModel.fit(events, settings=settings)

    assert a.head is None  # no correction head on the default path
    qa = np.vstack([a.q_all(c) for c in probe])
    qb = np.vstack([b.q_all(c) for c in probe])
    assert np.array_equal(qa, qb)  # deterministic, identical model


# --------------------------------------------------------------------------------------------- #
# On-ramp 1: decision-aware reweighting
# --------------------------------------------------------------------------------------------- #
def test_reweight_upweights_boundary_band() -> None:
    settings = Settings(seed=7)
    events = generate_logs(600, settings=settings, seed=7)
    labeled = [e for e in events if e.reward is not None]

    weights = decision_aware_weights(labeled, boundary_quantile=0.2, upweight=3.0)
    assert weights.shape == (len(labeled),)
    # Exactly two weight levels: the boundary band and the "obvious" rest.
    assert set(np.unique(weights).tolist()) == {1.0, 3.0}
    # Roughly the requested central fraction is upweighted (rank-quantile band of width 0.2).
    frac_up = float((weights > 1.0).mean())
    assert 0.12 < frac_up < 0.28
    # Rows near the median prize (boundary) are upweighted; the extremes are not.
    rewards = np.array([e.reward for e in labeled], dtype=np.float64)
    order = np.argsort(rewards)
    assert weights[order[0]] == 1.0  # lowest prize => obvious skip
    assert weights[order[-1]] == 1.0  # highest prize => obvious include


def test_reweight_changes_the_fit() -> None:
    settings = Settings(seed=7)
    events = generate_logs(800, settings=settings, seed=7)
    probe = _probe_contexts(30)

    df = settings.model_copy(update={"use_decision_focused": True, "df_mode": "reweight"})
    off = RewardModel.fit(events, settings=settings)
    rew = RewardModel.fit(events, settings=df)
    assert rew.head is None  # reweight works through sample_weight, not a head
    q_off = np.vstack([off.q_all(c) for c in probe])
    q_rew = np.vstack([rew.q_all(c) for c in probe])
    assert not np.allclose(q_off, q_rew)


# --------------------------------------------------------------------------------------------- #
# On-ramp 2: SPO+ fine-tune
# --------------------------------------------------------------------------------------------- #
def _spo_settings(**overrides: object) -> Settings:
    base = {
        "seed": 7,
        "use_decision_focused": True,
        "df_mode": "spo",
        "spo_epochs": 4,
        "spo_neighborhood_size": 8,
        "spo_max_neighborhoods": 40,
        "spo_time_limit_s": 0.03,
        "spo_lr": 0.01,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_spo_zero_step_leaves_head_at_zero() -> None:
    settings = _spo_settings(spo_epochs=0)
    events = generate_logs(400, settings=settings, seed=7)
    model = RewardModel.fit(events, settings=settings)
    assert model.head is not None
    # No epochs => no steps => the head is exactly zero, so pre-calibration scores equal the base.
    assert np.count_nonzero(model.head) == 0


def test_spo_model_is_a_calibrated_qmodel(tmp_path) -> None:
    settings = _spo_settings()
    events = generate_logs(800, settings=settings, seed=7)
    model = RewardModel.fit(events, settings=settings)

    n_features = featurize(_probe_contexts(1)[0], ACTIONS[0]).shape[0]
    assert model.head is not None and model.head.shape == (n_features,)
    assert model.calibrator is not None  # calibration refit after the fine-tune
    assert np.any(model.head != 0.0)  # the fine-tune actually moved the head

    ctx = _probe_contexts(1)[0]
    q = model.q_all(ctx)
    assert q.shape == (len(ACTIONS),)
    assert np.all(np.isfinite(q))

    # Persisted with the head; reload reproduces predictions.
    model.save(tmp_path / "m")
    reloaded = RewardModel.load(tmp_path / "m")
    assert reloaded.head is not None
    assert np.allclose(model.q_all(ctx), reloaded.q_all(ctx))


# --------------------------------------------------------------------------------------------- #
# Headline: decision-aware reweighting lowers decision regret
#
# The router's include decision under capacity is a top-``m`` selection by prize, so we grade
# regret exactly (no solver noise): value lost when the *model's* top-``m`` doors differ from the
# *oracle's* top-``m``. We use a weak-base regime (few logs) where the LightGBM base still
# misranks boundary doors — exactly what decision-aware reweighting is designed to fix. (At scale
# the base is already a near-optimal ranker, so the win shrinks into the noise; that honest limit
# is documented in the decision journal, mirroring Phases 10-11.)
# --------------------------------------------------------------------------------------------- #
def _sample_neighborhood(k: int, seed: int) -> list[ProspectContext]:
    settings = Settings(seed=7)
    rng = np.random.default_rng(seed)
    ames = load_ames(settings, seed=seed)
    rows = ames.sample(n=k, replace=True, random_state=seed).reset_index(drop=True)
    return [sample_context(rows.iloc[i].to_dict(), rng) for i in range(k)]


def _top_m(prizes: np.ndarray, m: int) -> np.ndarray:
    z = np.zeros(prizes.shape[0], dtype=np.float64)
    z[np.argsort(prizes)[-m:]] = 1.0
    return z


def _selection_regret(model: RewardModel, *, n_hoods: int, k: int, seed0: int = 5000) -> float:
    """Mean oracle value lost when servicing the model's top-``k//2`` doors vs the oracle's."""
    total = 0.0
    capacity = k // 2
    for h in range(n_hoods):
        ctxs = _sample_neighborhood(k, seed0 + h)
        oracle_prize = np.array([max(true_reward(c, a) for a in ACTIONS) for c in ctxs])
        model_prize = np.array([float(model.q_all(c).max()) for c in ctxs])
        served_oracle = oracle_prize @ _top_m(oracle_prize, capacity)
        served_model = oracle_prize @ _top_m(model_prize, capacity)
        total += float(served_oracle - served_model)
    return total / n_hoods


@pytest.mark.slow
def test_reweighting_lowers_decision_regret() -> None:
    settings = Settings(seed=7)
    events = generate_logs(300, settings=settings, seed=7)  # weak base => boundary doors misranked

    df = settings.model_copy(
        update={
            "use_decision_focused": True,
            "df_mode": "reweight",
            "df_boundary_quantile": 0.2,
            "df_upweight": 4.0,
        }
    )
    off = RewardModel.fit(events, settings=settings)
    rew = RewardModel.fit(events, settings=df)

    r_off = _selection_regret(off, n_hoods=50, k=16)
    r_rew = _selection_regret(rew, n_hoods=50, k=16)
    assert r_rew < r_off  # reweighting reallocates capacity to the boundary and wins the decision

    # Calibration survives: mean calibrated prediction still tracks realized mean reward, so the
    # decision win is not bought by wrecking the value estimate the OPE gate relies on.
    val = generate_logs(1500, settings=settings, seed=99)
    x = np.vstack([featurize(e.context, e.action) for e in val])
    y = np.array([e.reward for e in val], dtype=np.float64)
    assert rew._predict(x).mean() == pytest.approx(y.mean(), abs=0.05)


# --------------------------------------------------------------------------------------------- #
# The DF model is just another candidate through the existing OPE promotion gate
# --------------------------------------------------------------------------------------------- #
def test_df_model_runs_through_promotion_gate() -> None:
    from nba.bandits.epsilon_greedy import EpsilonGreedy
    from nba.ope.estimators import LoggedBatch, q_matrix
    from nba.ope.gate import PromotionGate

    settings = Settings(seed=7)
    events = generate_logs(1200, settings=settings, seed=7)
    model = RewardModel.fit(
        events,
        settings=settings.model_copy(update={"use_decision_focused": True, "df_mode": "reweight"}),
    )

    batch = LoggedBatch.from_events([e for e in events if e.reward is not None][:400])
    q_hat = q_matrix(model, batch.contexts)
    policy = EpsilonGreedy(model, epsilon=settings.epsilon, rng=np.random.default_rng(0))
    gate = PromotionGate(z=settings.ope_z, min_lift=settings.ope_min_lift)
    decision = gate.evaluate(policy, batch, q_hat, baseline_value=float(batch.rewards.mean()))

    assert np.isfinite(decision.candidate["dr"].value)
    assert np.isfinite(decision.lower_bound)
