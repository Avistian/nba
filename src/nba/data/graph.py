"""Turn a :class:`~nba.data.relational_simulator.RelationalWorld` into a typed heterogeneous graph.

The builder is gated by a **graph allow-list** that mirrors
:data:`nba.data.features.ALLOWED_FEATURES`: ``lat``/``lon``/``address_id`` and any protected field
are excluded *by construction*, so a forbidden attribute can enter neither a node feature nor
propagate via an edge.

There is **no torch dependency** here: :func:`build_graph` returns plain numpy arrays and index
tensors. The optional ``torch_geometric`` conversion lives in a later phase, so this module adds
zero heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nba.data.features import ALLOWED_FEATURES
from nba.data.relational_simulator import RelationalWorld

#: Per node type, the allow-listed fields that may become node features. Door features mirror the
#: model allow-list exactly; household nodes carry only the structural door count. Geo/identity and
#: any protected field are intentionally absent.
GRAPH_NODE_FEATURES: dict[str, tuple[str, ...]] = {
    "door": ALLOWED_FEATURES,
    "household": ("n_doors",),
}

#: The typed edge relations emitted by :func:`build_graph`.
GRAPH_EDGE_TYPES: tuple[str, ...] = ("near", "same_household", "shares_competitor", "interacted")

_NODE_TYPES: tuple[str, ...] = ("door", "household")


@dataclass(frozen=True)
class HeteroGraph:
    """A typed, numpy-backed heterogeneous graph.

    - ``node_features[type]`` is an ``(n_nodes, n_feats)`` float array.
    - ``edge_index[edge_type]`` is a ``(2, n_edges)`` int array of node positions. For door-door
      relations both rows index ``node_ids["door"]``; for ``interacted`` row 0 indexes a door and
      row 1 indexes a household.
    - ``node_ids[type]`` lists the stable string ids in row order.
    """

    node_features: dict[str, np.ndarray]
    edge_index: dict[str, np.ndarray]
    node_ids: dict[str, list[str]]


def allowed_node_feature(node_type: str, field: str) -> bool:
    """Return whether ``field`` is an allow-listed feature for ``node_type``."""
    return field in GRAPH_NODE_FEATURES.get(node_type, ())


def _door_feature_row(world: RelationalWorld, address_id: str) -> list[float]:
    ctx = world.contexts[address_id]
    return [float(getattr(ctx, name)) for name in GRAPH_NODE_FEATURES["door"]]


def build_graph(world: RelationalWorld) -> HeteroGraph:
    """Assemble nodes/edges using ONLY :data:`GRAPH_NODE_FEATURES`.

    Deterministic: node ids are sorted, so repeated builds of the same world are identical.
    """
    door_ids = sorted(world.contexts)
    door_pos = {addr: i for i, addr in enumerate(door_ids)}
    household_ids = sorted(h.household_id for h in world.households)
    household_pos = {hid: i for i, hid in enumerate(household_ids)}

    door_feats = (
        np.array([_door_feature_row(world, addr) for addr in door_ids], dtype=np.float64)
        if door_ids
        else np.zeros((0, len(GRAPH_NODE_FEATURES["door"])), dtype=np.float64)
    )
    members = world._household_members  # noqa: SLF001 - intra-package structural access
    household_feats = (
        np.array([[float(len(members.get(hid, ())))] for hid in household_ids], dtype=np.float64)
        if household_ids
        else np.zeros((0, len(GRAPH_NODE_FEATURES["household"])), dtype=np.float64)
    )

    def door_door(edges: list[tuple[str, str]]) -> np.ndarray:
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        src = [door_pos[a] for a, _ in edges]
        dst = [door_pos[b] for _, b in edges]
        return np.array([src, dst], dtype=np.int64)

    interacted_src: list[int] = []
    interacted_dst: list[int] = []
    for addr in door_ids:
        if not world.histories.get(addr):
            continue
        hid = world._addr_household.get(addr)  # noqa: SLF001 - intra-package structural access
        if hid is not None and hid in household_pos:
            interacted_src.append(door_pos[addr])
            interacted_dst.append(household_pos[hid])
    interacted = (
        np.array([interacted_src, interacted_dst], dtype=np.int64)
        if interacted_src
        else np.zeros((2, 0), dtype=np.int64)
    )

    return HeteroGraph(
        node_features={"door": door_feats, "household": household_feats},
        edge_index={
            "near": door_door(world.near_edges),
            "same_household": door_door(world.household_edges),
            "shares_competitor": door_door(world.competitor_edges),
            "interacted": interacted,
        },
        node_ids={"door": door_ids, "household": household_ids},
    )


def save_graph(graph: HeteroGraph, path: Path | str) -> None:
    """Serialize a :class:`HeteroGraph` to a compressed ``.npz`` archive."""
    arrays: dict[str, np.ndarray] = {
        "node_types": np.array(_NODE_TYPES, dtype=object),
        "edge_types": np.array(GRAPH_EDGE_TYPES, dtype=object),
    }
    for ntype in _NODE_TYPES:
        arrays[f"nodefeat__{ntype}"] = graph.node_features[ntype]
        arrays[f"nodeids__{ntype}"] = np.array(graph.node_ids[ntype], dtype=object)
    for etype in GRAPH_EDGE_TYPES:
        arrays[f"edge__{etype}"] = graph.edge_index[etype]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # ``**arrays`` carries only named feature/index arrays; pyright otherwise tries to bind them to
    # ``savez_compressed``'s ``allow_pickle`` keyword.
    np.savez_compressed(path, **arrays)  # pyright: ignore[reportArgumentType]


def load_graph(path: Path | str) -> HeteroGraph:
    """Load a :class:`HeteroGraph` previously written by :func:`save_graph`."""
    with np.load(path, allow_pickle=True) as data:
        node_types = [str(t) for t in data["node_types"]]
        edge_types = [str(t) for t in data["edge_types"]]
        node_features = {t: data[f"nodefeat__{t}"] for t in node_types}
        node_ids = {t: [str(x) for x in data[f"nodeids__{t}"]] for t in node_types}
        edge_index = {e: data[f"edge__{e}"] for e in edge_types}
    return HeteroGraph(node_features=node_features, edge_index=edge_index, node_ids=node_ids)
