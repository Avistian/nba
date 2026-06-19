"""Demo: you propose today's door list; the system plans the route and recommends at each stop.

Generates fresh simulator logs, trains a reward model in a throwaway sandbox, builds YOUR candidate
door pool near a chosen center, runs ``plan_route``, then walks the first N stops with
``recommend → feedback``.

Usage:
    uv run python scripts/my_territory_demo.py
    uv run python scripts/my_territory_demo.py --dataset relational --candidates 80 --walk 10
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from nba.api.store import EventStore
from nba.bandits.epsilon_greedy import EpsilonGreedy
from nba.config import Settings
from nba.data import relational_simulator as rel
from nba.data.ames import load_ames
from nba.data.simulator import generate_logs, sample_context
from nba.ethics import EthicalPolicy
from nba.eval.oracle import oracle_for
from nba.pipeline.orchestrator import Orchestrator
from nba.reward.model import RewardModel
from nba.routing.distance import HaversineEngine
from nba.schema import REWARD, ProspectContext


def _propose_doors(
    n: int,
    center: tuple[float, float],
    *,
    settings: Settings,
    seed: int,
    radius_km: float = 0.25,
) -> list[ProspectContext]:
    """Build the caller-owned candidate pool: fresh contexts placed near ``center``."""
    rng = np.random.default_rng(seed)
    ames = load_ames(settings, seed=seed)
    rows = ames.sample(n=n * 3, replace=True, random_state=seed).reset_index(drop=True)
    raw = [sample_context(rows.iloc[i].to_dict(), rng) for i in range(len(rows))]

    lat_km = 1.0 / 111.2
    lon_km = 1.0 / (111.2 * np.cos(np.radians(center[0])))
    placed: list[ProspectContext] = []
    for ctx in raw:
        r = radius_km * np.sqrt(rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        lat = center[0] + r * np.sin(theta) * lat_km
        lon = center[1] + r * np.cos(theta) * lon_km
        placed.append(ctx.model_copy(update={"lat": lat, "lon": lon}))

    rng.shuffle(placed)
    return placed[:n]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Territory demo: you propose doors, system routes."
    )
    parser.add_argument("--dataset", choices=["flat", "relational"], default="relational")
    parser.add_argument("--n-logs", type=int, default=8_000)
    parser.add_argument(
        "--candidates", type=int, default=60, help="doors YOU propose for the route"
    )
    parser.add_argument("--walk", type=int, default=8, help="stops to simulate door-by-door")
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center-lat", type=float, default=42.031)
    parser.add_argument("--center-lon", type=float, default=-93.618)
    args = parser.parse_args()

    center = (args.center_lat, args.center_lon)
    tmp = Path(tempfile.mkdtemp(prefix="nba_territory_"))
    settings = Settings(
        data_dir=tmp / "data",
        model_dir=tmp / "models",
        db_path=tmp / "events.db",
        dataset_mode=args.dataset,
        seed=args.seed,
    )
    settings.ensure_dirs()
    rng = np.random.default_rng(args.seed)

    print(f"sandbox     : {tmp}")
    print(f"generating  : {args.n_logs:,} logs ({args.dataset}) …")
    if settings.dataset_mode == "relational":
        events, _ = rel.generate_logs(args.n_logs, settings=settings, seed=args.seed)
    else:
        events = generate_logs(args.n_logs, settings=settings, seed=args.seed)

    model = RewardModel.fit(events, settings=settings)
    model.save(settings.model_dir)
    print(f"trained     : {settings.model_dir}")

    my_doors = _propose_doors(
        args.candidates, center, settings=settings, seed=args.seed + 1
    )
    print(f"you proposed: {len(my_doors)} doors near ({center[0]:.4f}, {center[1]:.4f})")

    policy = EthicalPolicy(
        EpsilonGreedy(model, epsilon=settings.epsilon, rng=rng),
        settings,
        rng=rng,
    )
    store = EventStore(settings.db_path)
    orch = Orchestrator(
        policy=policy,
        reward_model=model,
        distance_engine=HaversineEngine(speed_kmh=settings.walking_speed_kmh),
        store=store,
        settings=settings,
    )

    if settings.dataset_mode == "relational":
        eval_world = rel.world_from_contexts(my_doors, settings=settings, seed=args.seed + 2)
    else:
        eval_world = None
    oracle = oracle_for(settings, world=eval_world)

    route = orch.plan_route(my_doors)
    print(
        f"route       : {len(route.visited)} visited, {len(route.dropped)} dropped, "
        f"{route.total_time_s / 60:.1f} min, profit {route.total_profit:.2f}"
    )

    remaining = list(my_doors)
    order = [remaining[i - 1] for i in route.order if i != 0]
    step = 0
    while order and step < args.walk:
        door = order.pop(0)
        rec = orch.recommend(door)
        outcome = oracle.sample_outcome(door, rec.action, rng)
        orch.feedback(rec.decision_id, outcome)
        print(
            f"  stop {step + 1:2d}: {rec.action.value:14s}  p={rec.propensity:.3f}  "
            f"→ {outcome.value:12s}  r={REWARD[outcome]:+.2f}"
        )
        remaining.remove(door)
        step += 1
        if step % args.replan_every == 0 and remaining and order:
            route = orch.replan(remaining)
            order = [remaining[i - 1] for i in route.order if i != 0]
            print(f"  replanned  : {len(route.visited)} stops remaining")

    print(f"logged      : {orch.decision_count()} decisions in {settings.db_path}")


if __name__ == "__main__":
    main()
