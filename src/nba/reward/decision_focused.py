"""Decision-focused learning: train the reward model to make good *routes*, not accurate numbers.

Today's pipeline is predict-then-optimize: the :class:`~nba.reward.model.RewardModel` minimizes
squared error, then the router consumes its scores. But those scores are only ever used to make
include/skip/order decisions, so squared error spends capacity on doors whose value is so high or
so low that the routing decision is obvious and a wrong prediction costs nothing. This module
shifts the objective toward the doors *where being wrong changes the route*.

Two on-ramps, both behind ``Settings.use_decision_focused`` (off by default):

- :func:`decision_aware_weights` (``df_mode="reweight"``) — a cheap, gradient-free approximation
  that upweights training rows near the historical include/skip boundary. Flows into LightGBM's
  ``sample_weight``.
- :func:`spo_finetune` (``df_mode="spo"``) — the real thing: an SPO+ (Smart Predict-then-Optimize)
  subgradient loop (Elmachtoub & Grigas 2021) that fits a linear correction *head* on top of the
  frozen booster so the route the model induces matches the route the realized rewards would induce.

**Safety rails (enforced by the ethics AST guard over ``src/nba/reward/``):** no simulator oracle is
imported or referenced. SPO+'s "true prize" is the **realized logged reward** ``event.reward`` — the
production-faithful label the method is designed for — never ``true_reward``/``true_best_action``.
The isotonic calibrator is refit after fine-tuning so DM/DR off-policy evaluation stays valid.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from sklearn.isotonic import IsotonicRegression

from nba.config import Settings
from nba.data.features import featurize, featurize_batch
from nba.routing.distance import HaversineEngine
from nba.routing.tsp_profits import Route, solve_tsp_profits
from nba.schema import BanditEvent

if TYPE_CHECKING:
    from nba.reward.model import RewardModel


def decision_aware_weights(
    events: Sequence[BanditEvent], *, boundary_quantile: float, upweight: float
) -> np.ndarray:
    """Per-row training weight: upweight rows near the historical include/skip boundary.

    A door whose logged reward sits in the middle of the prize distribution is one the router could
    plausibly include *or* skip — its prediction is decision-relevant. Doors at the extremes are
    "obvious" includes/skips whose routing decision a wrong prediction rarely flips. We therefore
    scale the central band ``[0.5 - q/2, 0.5 + q/2]`` (in prize-quantile space) by ``upweight`` and
    leave the rest at ``1.0``. Gradient-free and A/B-able immediately (doc 16 section 3).

    Args:
        events: labeled logged events (``reward`` must be set).
        boundary_quantile: width ``q`` of the upweighted central band, in ``(0, 1]``.
        upweight: multiplier applied to boundary-band rows (``>= 1`` to upweight).

    Returns:
        A ``float64`` array of length ``len(events)`` of per-row sample weights.
    """
    if not 0.0 < boundary_quantile <= 1.0:
        raise ValueError(f"boundary_quantile must be in (0, 1], got {boundary_quantile}")
    rewards = np.array([e.reward for e in events], dtype=np.float64)
    n = rewards.size
    if n == 0:
        return np.ones(0, dtype=np.float64)

    # Per-row rank quantile in (0, 1). Using row ranks (not value-collapsed ranks) guarantees the
    # central band is populated even on the discrete reward scale, so ~``boundary_quantile`` of rows
    # — those nearest the median prize, i.e. the include/skip margin — are upweighted.
    order = np.argsort(rewards, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    quantile = (ranks + 0.5) / n

    half = boundary_quantile / 2.0
    in_band = np.abs(quantile - 0.5) <= half
    weights = np.ones(n, dtype=np.float64)
    weights[in_band] = float(upweight)
    return weights


def _centroid(coords: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the ``(lat, lon)`` centroid of a set of door coordinates (the stand-in depot)."""
    lats = [lat for lat, _ in coords]
    lons = [lon for _, lon in coords]
    return float(np.mean(lats)), float(np.mean(lons))


def _dense_coords(
    n: int,
    rng: np.random.Generator,
    *,
    center: tuple[float, float] = (42.03, -93.62),
    radius_km: float = 0.4,
) -> list[tuple[float, float]]:
    """Place ``n`` synthetic doors uniformly on a walkable block for the SPO+ inner routes.

    The logged simulator scatters doors over kilometres, so raw coordinates make travel dominate the
    route and the include/skip decision ignores prize — no learning signal. Repositioning doors onto
    a dense block (exactly what ``run_demo._dense_block`` does for the shift) makes prize and travel
    comparable, so SPO+ trains through the *real* prize/travel trade-off. Geometry is safe to
    synthesize: ``lat``/``lon`` never enter a model (the ethics allow-list excludes them).
    """
    lat_deg_per_km = 1.0 / 111.2
    lon_deg_per_km = 1.0 / (111.2 * np.cos(np.radians(center[0])))
    r = radius_km * np.sqrt(rng.random(n))  # sqrt -> uniform over the disk
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    lats = center[0] + r * np.sin(theta) * lat_deg_per_km
    lons = center[1] + r * np.cos(theta) * lon_deg_per_km
    return list(zip(lats.tolist(), lons.tolist(), strict=True))


def _include_vector(
    door_coords: list[tuple[float, float]],
    prizes: np.ndarray,
    *,
    settings: Settings,
    engine: HaversineEngine,
) -> np.ndarray:
    """Solve the small prize-collecting route and return a 0/1 include indicator per door.

    A binding capacity (half the neighborhood) forces the solver to *choose* which doors to service,
    so the include/skip decision — the thing SPO+ shapes — is actually exercised. Non-positive
    prizes are dropped for free (the solver floors the drop penalty at zero).
    """
    k = len(door_coords)
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    coords = [_centroid(door_coords), *door_coords]
    profits = [0.0, *prizes.tolist()]
    time_matrix = engine.time_matrix(coords)
    capacity = max(1, k // 2)
    route = solve_tsp_profits(
        coords,
        profits,
        time_matrix,
        depot=0,
        capacity=capacity,
        drop_scale=settings.drop_scale,
        lambda_travel=settings.lambda_travel,
        time_limit_s=settings.spo_time_limit_s,
        seed=settings.seed,
    )
    assert isinstance(route, Route)  # num_vehicles defaults to 1 -> a single Route
    z = np.zeros(k, dtype=np.float64)
    for node in route.visited:
        z[node - 1] = 1.0  # coords[0] is the depot, so door j is node j + 1
    return z


def spo_finetune(
    model: RewardModel, events: Sequence[BanditEvent], *, settings: Settings
) -> RewardModel:
    """Fine-tune a linear correction head on ``model`` via the SPO+ subgradient over logged routes.

    The head ``w`` (over the frozen ``featurize(x, a)`` vector) starts at zero, so the returned
    model begins identical to ``model`` and moves only toward better routing decisions. For each
    synthetic neighborhood (a random group of logged doors) we solve two prize-collecting routes —
    one priced by the realized rewards ``c`` and one by the SPO+ perturbation ``2*c_hat - c`` — and
    step ``w`` down the resulting subgradient ``2 * (z_spo - z_true)`` chained through the feature
    rows. The isotonic calibrator is refit afterward on the corrected scores so DM/DR OPE stays
    valid.

    The returned object is still a :class:`~nba.reward.model.RewardModel` (a ``QModel``); no serving
    interface changes.
    """
    from nba.reward.model import RewardModel  # local import avoids a module import cycle

    labeled = [e for e in events if e.reward is not None]
    n = len(labeled)
    # A door's routing prize is its *best-action* value ``max_a q(x, a)`` — the quantity the router
    # collects — so SPO+ must correct that, not the logged action alone. We therefore carry the full
    # per-action feature stack and let the argmax pick the responsible action per door each step.
    phi_actions = np.stack([featurize_batch(e.context) for e in labeled])  # (n, |A|, F)
    n_features = phi_actions.shape[2]
    base_actions = np.asarray(
        model.booster.predict(phi_actions.reshape(-1, n_features)), dtype=np.float64
    ).reshape(n, -1)  # (n, |A|)
    phi_logged = np.vstack([featurize(e.context, e.action) for e in labeled])  # for calibration
    base_logged = np.asarray(model.booster.predict(phi_logged), dtype=np.float64)
    c_true = np.array([e.reward for e in labeled], dtype=np.float64)

    # Standardize features for the *linear* head. The booster is scale-invariant, but raw features
    # (e.g. property_value ~ 1e5) would make a linear subgradient explode. We fit ``w`` in
    # standardized space, then fold the scale back into a raw-space head so ``_predict`` stays a
    # plain ``booster + x @ head``; the dropped constant offset shifts every score equally, so it
    # changes no include/skip/argmax decision and is reabsorbed by the refit calibrator.
    mu = phi_logged.mean(axis=0)
    sigma = phi_logged.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    phi_actions_std = (phi_actions - mu) / sigma  # (n, |A|, F)

    rng = np.random.default_rng(settings.seed)
    # Synthetic walkable geometry for the training routes (see _dense_coords): prize and travel are
    # comparable so the include/skip decision — the thing SPO+ shapes — actually depends on prize.
    coords_all = _dense_coords(n, rng)
    perm = rng.permutation(n)
    size = max(2, settings.spo_neighborhood_size)
    groups = [perm[i : i + size] for i in range(0, n, size)]
    groups = [g for g in groups if g.size >= 2]
    if len(groups) > settings.spo_max_neighborhoods:
        groups = groups[: settings.spo_max_neighborhoods]

    w = np.zeros(n_features, dtype=np.float64)
    if groups:
        engine = HaversineEngine(speed_kmh=settings.walking_speed_kmh)
        rows = np.arange(size)
        for _ in range(settings.spo_epochs):
            epoch_order = rng.permutation(len(groups))
            for start in range(0, epoch_order.size, settings.spo_batch):
                batch = epoch_order[start : start + settings.spo_batch]
                grad = np.zeros(n_features, dtype=np.float64)
                for gi in batch:
                    members = groups[gi]
                    k = members.size
                    phi_nb = phi_actions_std[members]  # (k, |A|, F) standardized
                    scores = base_actions[members] + phi_nb @ w  # (k, |A|)
                    a_star = np.argmax(scores, axis=1)  # responsible action per door
                    c_hat = scores[rows[:k], a_star]  # (k,) best-action prize
                    grad_feats = phi_nb[rows[:k], a_star]  # (k, F) subgradient of the max
                    c = c_true[members]
                    door_coords = [coords_all[j] for j in members]
                    z_true = _include_vector(door_coords, c, settings=settings, engine=engine)
                    z_spo = _include_vector(
                        door_coords, 2.0 * c_hat - c, settings=settings, engine=engine
                    )
                    subgrad_doors = 2.0 * (z_spo - z_true)  # dL/dc_hat per door
                    grad += grad_feats.T @ subgrad_doors  # chain through c_hat = base + w . phi
                grad /= max(1, batch.size)
                w -= settings.spo_lr * (grad + settings.spo_l2 * w)

    # Fold standardization back into a raw-space head: (x - mu)/sigma . w == x . (w/sigma) + const.
    head = w / sigma

    # Refit the isotonic calibrator on the corrected logged-action scores (the exact quantity
    # ``_predict`` produces) so DM/DR stay calibrated.
    val_rng = np.random.default_rng(settings.seed)
    val_perm = val_rng.permutation(n)
    n_val = max(1, int(n * 0.2))
    val_idx = val_perm[:n_val]
    adjusted = base_logged[val_idx] + phi_logged[val_idx] @ head
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(adjusted, c_true[val_idx])

    return RewardModel(
        booster=model.booster,
        calibrator=calibrator,
        feature_names=model.feature_names,
        head=head,
    )
