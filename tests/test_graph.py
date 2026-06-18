"""Tests for the heterogeneous-graph builder: declared types, allow-list, determinism, I/O."""

from __future__ import annotations

import numpy as np

from nba.config import Settings
from nba.data.features import ALLOWED_FEATURES
from nba.data.graph import (
    GRAPH_EDGE_TYPES,
    GRAPH_NODE_FEATURES,
    allowed_node_feature,
    build_graph,
    load_graph,
    save_graph,
)
from nba.data.relational_simulator import sample_world

_FORBIDDEN = {"lat", "lon", "address_id", "age", "race", "gender", "ethnicity", "zip"}


def _world(n: int = 400):
    return sample_world(n, settings=Settings(), seed=7)


def test_graph_has_declared_node_and_edge_types() -> None:
    graph = build_graph(_world())
    assert set(graph.node_features) == {"door", "household"}
    assert set(graph.edge_index) == set(GRAPH_EDGE_TYPES)
    assert all(graph.edge_index[e].shape[0] == 2 for e in GRAPH_EDGE_TYPES)


def test_counts_match_world() -> None:
    world = _world()
    graph = build_graph(world)
    assert len(graph.node_ids["door"]) == len(world.contexts)
    assert len(graph.node_ids["household"]) == len(world.households)
    assert graph.node_features["door"].shape == (len(world.contexts), len(ALLOWED_FEATURES))
    assert graph.edge_index["near"].shape[1] == len(world.near_edges)
    assert graph.edge_index["same_household"].shape[1] == len(world.household_edges)
    assert graph.edge_index["shares_competitor"].shape[1] == len(world.competitor_edges)


def test_allow_list_excludes_geo_identity_and_protected() -> None:
    assert not (_FORBIDDEN & set(GRAPH_NODE_FEATURES["door"]))
    assert set(GRAPH_NODE_FEATURES["door"]) == set(ALLOWED_FEATURES)
    assert not allowed_node_feature("door", "lat")
    assert not allowed_node_feature("door", "address_id")
    assert allowed_node_feature("door", "property_value")


def test_deterministic_by_seed() -> None:
    g1 = build_graph(_world())
    g2 = build_graph(_world())
    assert g1.node_ids == g2.node_ids
    assert np.array_equal(g1.node_features["door"], g2.node_features["door"])
    for etype in GRAPH_EDGE_TYPES:
        assert np.array_equal(g1.edge_index[etype], g2.edge_index[etype])


def test_save_load_round_trip(tmp_path) -> None:
    graph = build_graph(_world())
    path = tmp_path / "graph.npz"
    save_graph(graph, path)
    loaded = load_graph(path)

    assert loaded.node_ids == graph.node_ids
    for ntype in graph.node_features:
        assert np.array_equal(loaded.node_features[ntype], graph.node_features[ntype])
    for etype in GRAPH_EDGE_TYPES:
        assert np.array_equal(loaded.edge_index[etype], graph.edge_index[etype])
