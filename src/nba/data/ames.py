"""Ames-housing-backed prospect features, with an offline synthetic fallback.

The real Ames dataset gives realistic property values, build years, and lot sizes. When the
download is unavailable (offline CI, no network), :func:`synthetic_ames` produces a statistically
similar frame so the rest of the pipeline is fully testable without the internet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba.config import Settings

#: Public mirror of the Ames housing dataset (De Cock, 2011).
AMES_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"  # documented source
)

#: Columns the rest of the pipeline relies on, after normalization.
AMES_COLUMNS: tuple[str, ...] = ("sale_price", "year_built", "lot_area")

#: Cache filename under ``settings.data_dir``.
_CACHE_NAME = "ames.parquet"


def synthetic_ames(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return an ``n``-row synthetic stand-in for the Ames dataset.

    ``sale_price`` is log-normal, ``year_built`` roughly uniform over the last century, and
    ``lot_area`` log-normal and mildly correlated with price.
    """
    sale_price = rng.lognormal(mean=12.1, sigma=0.35, size=n)  # ~ $180k median
    year_built = rng.integers(1920, 2016, size=n).astype(np.float64)
    lot_area = rng.lognormal(mean=9.1, sigma=0.30, size=n) * (1.0 + 0.05 * rng.standard_normal(n))
    return pd.DataFrame(
        {
            "sale_price": sale_price,
            "year_built": year_built,
            "lot_area": np.abs(lot_area),
        }
    )


def _normalize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a raw Ames frame to :data:`AMES_COLUMNS`; tolerant of column-name variants."""
    lower = {c.lower(): c for c in raw.columns}

    def pick(*candidates: str) -> str | None:
        for cand in candidates:
            if cand in lower:
                return lower[cand]
        return None

    price = pick("saleprice", "sale_price", "medv", "price")
    year = pick("yearbuilt", "year_built", "yr_built")
    lot = pick("lotarea", "lot_area", "lot")
    if price is None:
        raise ValueError("Ames frame missing a recognizable sale-price column")

    out = pd.DataFrame()
    out["sale_price"] = pd.to_numeric(raw[price], errors="coerce")
    out["year_built"] = pd.to_numeric(raw[year], errors="coerce") if year else np.nan
    out["lot_area"] = pd.to_numeric(raw[lot], errors="coerce") if lot else np.nan
    out = out.dropna(subset=["sale_price"]).reset_index(drop=True)
    # MEDV-style mirrors are in $1000s; scale up so downstream USD math is sane.
    if out["sale_price"].median() < 1000:
        out["sale_price"] = out["sale_price"] * 1000.0
    return out


def _download_ames() -> pd.DataFrame:
    """Fetch and normalize the Ames dataset. Raises on any failure (caller falls back)."""
    raw = pd.read_csv(AMES_URL)
    return _normalize_raw(raw)


def load_ames(settings: Settings, *, n_fallback: int = 5000, seed: int = 7) -> pd.DataFrame:
    """Return the Ames frame, cached at ``data/ames.parquet``.

    Tries the local cache, then the network download, then a synthetic fallback. Never raises on
    a missing network: offline use is a first-class path.
    """
    cache = settings.data_dir / _CACHE_NAME
    if cache.exists():
        return pd.read_parquet(cache)

    try:
        frame = _download_ames()
    except Exception:
        frame = synthetic_ames(n_fallback, np.random.default_rng(seed))

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache)
    return frame
