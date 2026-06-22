"""Generate the relational dataset: logged bandit feedback + entity tables + a typed graph.

Mirrors ``scripts/generate_logs.py`` but uses the relational simulator, which carries genuine
relational/temporal ground truth (households, neighbor edges, histories, competitor overlap). The
emitted ``logs.parquet`` is schema-identical to the flat one (plus a non-model ``household_id``
column); the relational structure lands in sidecar artifacts consumed only by a future RDL model.

Usage:
    uv run python scripts/generate_relational_logs.py --n 20000 --seed 7 --out data/relational
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nba.config import get_settings
from nba.data.drift import generate_logs_for_settings
from nba.data.graph import build_graph, save_graph
from nba.data.relational_simulator import logs_to_frame


def _households_frame(households: list) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        {
            "household_id": h.household_id,
            "n_doors": len(h.address_ids),
            "centroid_lat": h.centroid[0],
            "centroid_lon": h.centroid[1],
            "address_ids": ",".join(h.address_ids),
        }
        for h in households
    )


def _edges_frame(world) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for kind, edges in (
        ("near", world.near_edges),
        ("same_household", world.household_edges),
        ("shares_competitor", world.competitor_edges),
    ):
        records.extend({"edge_type": kind, "src": a, "dst": b} for a, b in edges)
    return pd.DataFrame.from_records(records, columns=["edge_type", "src", "dst"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the relational NBA dataset.")
    parser.add_argument("--n", type=int, default=20_000, help="number of events/doors to generate")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--out", type=Path, default=Path("data/relational"), help="output directory"
    )
    parser.add_argument(
        "--temp", type=float, default=0.5, help="logging-policy softmax temperature"
    )
    args = parser.parse_args()

    settings = get_settings()
    events, world = generate_logs_for_settings(
        args.n, settings=settings, seed=args.seed, temp=args.temp
    )
    assert world is not None

    frame = logs_to_frame(events, world=world)
    graph = build_graph(world)

    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "logs.parquet")
    _households_frame(world.households).to_parquet(args.out / "households.parquet")
    _edges_frame(world).to_parquet(args.out / "edges.parquet")
    save_graph(graph, args.out / "graph.npz")

    arm_counts = frame["action"].value_counts().to_dict()
    n_edges = sum(ei.shape[1] for ei in graph.edge_index.values())
    print(f"wrote {len(frame)} events -> {args.out}/logs.parquet")
    print(f"arm frequencies:      {arm_counts}")
    print(f"mean reward:          {frame['reward'].mean():.4f}")
    print(f"min propensity:       {frame['propensity'].min():.4f} (> 0 required for OPE)")
    print(f"households:            {len(world.households)}")
    n_door = len(graph.node_ids["door"])
    n_hh = len(graph.node_ids["household"])
    print(f"nodes (door/hh):       {n_door}/{n_hh}")
    print(f"edges (all types):     {n_edges}")
    for etype, ei in graph.edge_index.items():
        print(f"  {etype:<18} {ei.shape[1]}")
    print(f"doors with a neighbor: {world.neighbor_fraction():.1%}")


if __name__ == "__main__":
    main()
