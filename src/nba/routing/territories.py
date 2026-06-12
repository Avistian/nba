"""Carve a flat list of doors into walkable territories via KMeans.

Pre-partitioning matters for two reasons: OR-Tools' routing solver scales poorly past a few
hundred nodes, and a single rep's shift should stay inside one geographically coherent cluster
rather than zig-zag across town. Clustering happens in a lightly rescaled lat/lon space so that
one degree of longitude counts the same as one degree of latitude (roughly equal-area), which
keeps clusters compact on the ground instead of stretched east-west.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class Territory:
    """A walkable cluster of doors.

    ``indices`` are positions into the *original* door list passed to
    :func:`cluster_territories`, so callers can map straight back to their data.
    """

    label: int
    indices: list[int]
    centroid: tuple[float, float]


def _scaled_coords(arr: np.ndarray) -> np.ndarray:
    """Rescale lon by ``cos(mean lat)`` so degrees are roughly equal-area before clustering."""
    mean_lat_rad = math.radians(float(arr[:, 0].mean()))
    lon_scale = math.cos(mean_lat_rad)
    return np.column_stack([arr[:, 0], arr[:, 1] * lon_scale])


def cluster_territories(
    coords: Sequence[tuple[float, float]], *, target_size: int, seed: int
) -> list[Territory]:
    """Partition ``coords`` into ``ceil(n / target_size)`` walkable territories.

    ``target_size`` is the desired number of doors per cluster (e.g. one rep's shift capacity).
    Clustering is deterministic given ``seed``.
    """
    if target_size < 1:
        raise ValueError("target_size must be >= 1")

    n = len(coords)
    if n == 0:
        return []

    arr = np.asarray(coords, dtype=np.float64)
    n_clusters = min(math.ceil(n / target_size), n)

    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(
        _scaled_coords(arr)
    )

    territories: list[Territory] = []
    for label in range(n_clusters):
        members = [i for i in range(n) if labels[i] == label]
        if not members:  # KMeans can, rarely, return an empty cluster.
            continue
        member_arr = arr[members]
        centroid = (float(member_arr[:, 0].mean()), float(member_arr[:, 1].mean()))
        territories.append(Territory(label=label, indices=members, centroid=centroid))
    return territories


def assign_doors(
    coords: Sequence[tuple[float, float]], territories: Sequence[Territory]
) -> dict[int, int]:
    """Invert :func:`cluster_territories` into a ``door_index -> territory label`` map."""
    mapping: dict[int, int] = {}
    for territory in territories:
        for door_index in territory.indices:
            mapping[door_index] = territory.label
    if len(mapping) != len(coords):
        raise ValueError(
            f"territories cover {len(mapping)} doors but {len(coords)} coords were given"
        )
    return mapping
