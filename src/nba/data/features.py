"""Deterministic feature engineering with an ethics allow-list.

``featurize(ctx, action)`` turns a context/action pair into a fixed-width ``float64`` vector. The
column order is frozen and exposed via :data:`FEATURE_NAMES`, the single source of truth that
models persist and verify on load.

Only fields named in :data:`ALLOWED_FEATURES` reach a model. Geo/identity fields (``lat``,
``lon``, ``address_id``) and any protected attribute are excluded by construction: the vector is
built solely from the allow-list, never by reflecting over the model.
"""

from __future__ import annotations

import numpy as np

from nba.schema import ACTIONS, Action, ProspectContext

#: Numeric/boolean context fields that may enter a model. Weather is one-hot expanded separately
#: (see :data:`WEATHER_LEVELS`); geo/identity fields are intentionally absent.
ALLOWED_FEATURES: tuple[str, ...] = (
    "property_value",
    "roof_age_years",
    "est_income",
    "tenure_years",
    "prior_interactions",
    "hour",
    "dow",
    "block_density",
    "neighbor_recent_conversion",
    "distance_from_rep_km",
    "nearby_high_reward_density",
)

#: Weather levels, frozen for one-hot encoding. Mirrors the ``weather`` ``Literal`` in the schema.
WEATHER_LEVELS: tuple[str, ...] = ("clear", "rain", "cold", "hot")

#: Frozen column order: allowed numeric/bool fields, then weather one-hot, then action one-hot.
FEATURE_NAMES: list[str] = [
    *ALLOWED_FEATURES,
    *(f"weather={level}" for level in WEATHER_LEVELS),
    *(f"action={action.value}" for action in ACTIONS),
]


def n_features() -> int:
    """Return the fixed feature-vector width."""
    return len(FEATURE_NAMES)


def context_vector(ctx: ProspectContext) -> np.ndarray:
    """Return the numeric/bool allow-list block followed by the weather one-hot block."""
    numeric = np.fromiter(
        (float(getattr(ctx, name)) for name in ALLOWED_FEATURES),
        dtype=np.float64,
        count=len(ALLOWED_FEATURES),
    )
    weather = np.zeros(len(WEATHER_LEVELS), dtype=np.float64)
    weather[WEATHER_LEVELS.index(ctx.weather)] = 1.0
    return np.concatenate([numeric, weather])


def action_onehot(action: Action) -> np.ndarray:
    """Return the one-hot encoding of ``action`` in canonical :data:`~nba.schema.ACTIONS` order."""
    vec = np.zeros(len(ACTIONS), dtype=np.float64)
    vec[ACTIONS.index(action)] = 1.0
    return vec


def featurize(ctx: ProspectContext, action: Action) -> np.ndarray:
    """Return the fixed-width ``float64`` feature vector for ``(ctx, action)``."""
    return np.concatenate([context_vector(ctx), action_onehot(action)])


def featurize_batch(ctx: ProspectContext, actions: tuple[Action, ...] = ACTIONS) -> np.ndarray:
    """Return a ``(len(actions), n_features())`` matrix, one featurized row per action."""
    ctx_block = context_vector(ctx)
    rows = [np.concatenate([ctx_block, action_onehot(action)]) for action in actions]
    return np.vstack(rows)
