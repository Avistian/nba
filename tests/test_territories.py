"""Tests for KMeans territory clustering."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from nba.routing.territories import Territory, assign_doors, cluster_territories


def _blob(
    center: tuple[float, float], n: int, rng: np.random.Generator
) -> list[tuple[float, float]]:
    lat, lon = center
    return [(lat + rng.normal(0, 0.002), lon + rng.normal(0, 0.002)) for _ in range(n)]


def test_cluster_count_and_full_coverage() -> None:
    rng = np.random.default_rng(0)
    coords = _blob((42.0, -93.6), 50, rng)

    territories = cluster_territories(coords, target_size=10, seed=7)

    assert len(territories) == math.ceil(50 / 10)
    counts = Counter(i for t in territories for i in t.indices)
    assert sorted(counts) == list(range(50))
    assert all(v == 1 for v in counts.values())  # each door assigned exactly once


def test_cluster_is_deterministic_by_seed() -> None:
    rng = np.random.default_rng(1)
    coords = _blob((42.0, -93.6), 40, rng)

    a = cluster_territories(coords, target_size=8, seed=7)
    b = cluster_territories(coords, target_size=8, seed=7)

    assert [t.indices for t in a] == [t.indices for t in b]


def test_nearby_doors_share_a_label() -> None:
    rng = np.random.default_rng(2)
    near = _blob((42.0, -93.6), 10, rng)
    far = _blob((42.5, -93.0), 10, rng)
    coords = near + far

    territories = cluster_territories(coords, target_size=10, seed=7)
    assignment = assign_doors(coords, territories)

    near_labels = {assignment[i] for i in range(10)}
    far_labels = {assignment[i] for i in range(10, 20)}
    assert len(near_labels) == 1
    assert len(far_labels) == 1
    assert near_labels != far_labels


def test_empty_coords_yield_no_territories() -> None:
    assert cluster_territories([], target_size=10, seed=7) == []


def test_target_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        cluster_territories([(0.0, 0.0)], target_size=0, seed=7)


def test_assign_doors_rejects_incomplete_cover() -> None:
    territory = Territory(label=0, indices=[0], centroid=(0.0, 0.0))
    with pytest.raises(ValueError):
        assign_doors([(0.0, 0.0), (1.0, 1.0)], [territory])
