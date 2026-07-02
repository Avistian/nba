"""End-to-end, fully-offline demo of the whole NBA loop for one simulated shift.

Pipeline (all seeded):

1. Generate logged feedback from the simulator and split it train / held-out.
2. Fit the calibrated reward model on the train split.
3. Build all three bandit policies (epsilon-greedy, UCB, Thompson).
4. Off-policy-evaluate each on the held-out split; the promotion gate picks the best policy that
   beats the logging baseline.
5. Simulate a shift over a dense neighborhood of fresh doors: plan a route, walk it (recommend ->
   simulate outcome -> feedback), replanning every few doors.
6. Compare the selected bandit against uniform-random and exploit-only baselines, measure regret
   against the oracle, and quantify routing savings vs a visit-all nearest-neighbor tour.

The report is printed and written to ``artifacts/demo_report.json``. The simulator oracle
(``true_reward`` etc.) is used here only for *evaluation*, exactly as in tests/notebooks.

Usage:
    uv run python scripts/run_demo.py --n-logs 20000 --shift 60 --replan-every 10 --seed 7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nba.bandits.base import Policy
from nba.bandits.epsilon_greedy import EpsilonGreedy
from nba.bandits.thompson import BootstrapEnsemble, ThompsonSampling
from nba.bandits.ucb import UCB
from nba.config import Settings, get_settings
from nba.data import relational_simulator as rel
from nba.data.ames import load_ames
from nba.data.drift import generate_logs_for_settings
from nba.data.simulator import sample_context
from nba.ethics import EthicalPolicy
from nba.eval.oracle import oracle_for
from nba.ope.estimators import LoggedBatch, q_matrix
from nba.ope.gate import PromotionGate
from nba.pipeline.orchestrator import Orchestrator, build_distance_engine
from nba.reward.model import RewardModel
from nba.routing.tsp_profits import Route
from nba.schema import ACTIONS, REWARD, ProspectContext


@dataclass
class DemoReport:
    """The structured outcome of a demo run (the JSON-safe part is written to disk)."""

    seed: int
    n_logs: int
    shift: int
    replan_every: int
    baseline_value: float
    selected_policy: str
    gate_promoted: bool
    selected_dr_lower_bound: float
    ope: dict[str, dict[str, float]]
    expected_reward: dict[str, float]
    realized_reward_bandit: float
    regret_curve: list[float]
    avg_regret_curve: list[float]
    routing: dict[str, float]
    n_decisions: int
    min_propensity: float
    # Runtime-only handles for tests/notebooks; excluded from JSON.
    model: RewardModel | None = field(default=None, repr=False)
    policy: Policy | None = field(default=None, repr=False)
    settings: Settings | None = field(default=None, repr=False)

    def to_json(self) -> dict[str, Any]:
        """Return the JSON-serializable subset of the report."""
        return {
            "seed": self.seed,
            "n_logs": self.n_logs,
            "shift": self.shift,
            "replan_every": self.replan_every,
            "baseline_value": self.baseline_value,
            "selected_policy": self.selected_policy,
            "gate_promoted": self.gate_promoted,
            "selected_dr_lower_bound": self.selected_dr_lower_bound,
            "ope": self.ope,
            "expected_reward": self.expected_reward,
            "realized_reward_bandit": self.realized_reward_bandit,
            "regret_curve": self.regret_curve,
            "avg_regret_curve": self.avg_regret_curve,
            "routing": self.routing,
            "n_decisions": self.n_decisions,
            "min_propensity": self.min_propensity,
        }


def _haversine_km(center: np.ndarray, pts: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    lat1, lon1 = np.radians(center)
    lat2, lon2 = np.radians(pts[:, 0]), np.radians(pts[:, 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _sample_contexts(n: int, settings: Settings, seed: int) -> list[ProspectContext]:
    rng = np.random.default_rng(seed)
    ames = load_ames(settings, seed=seed)
    rows = ames.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    return [sample_context(rows.iloc[i].to_dict(), rng) for i in range(n)]


def _dense_block(
    n: int,
    settings: Settings,
    seed: int,
    *,
    center: tuple[float, float] = (42.03, -93.62),
    radius_km: float = 0.3,
) -> list[ProspectContext]:
    """Sample ``n`` contexts and place them on a walkable block of the given radius.

    The simulator scatters doors over several km — fine for *who* lives there, but not a route a
    rep could walk. Since ``lat``/``lon`` are excluded from every model by the ethics allow-list,
    repositioning doors onto a dense block changes geography only, never reward, profit, or outcome.
    """
    contexts = _sample_contexts(n, settings, seed)
    rng = np.random.default_rng(seed)
    lat_deg_per_km = 1.0 / 111.2
    lon_deg_per_km = 1.0 / (111.2 * np.cos(np.radians(center[0])))

    placed: list[ProspectContext] = []
    for ctx in contexts:
        r = radius_km * np.sqrt(rng.random())  # sqrt → uniform over the disk
        theta = rng.uniform(0.0, 2.0 * np.pi)
        lat = center[0] + r * np.sin(theta) * lat_deg_per_km
        lon = center[1] + r * np.cos(theta) * lon_deg_per_km
        placed.append(ctx.model_copy(update={"lat": lat, "lon": lon}))
    return placed


def _build_policies(
    model: RewardModel, train_events: list, settings: Settings, rng: np.random.Generator
) -> list[Policy]:
    ensemble = BootstrapEnsemble.fit(train_events, settings=settings, n_models=settings.n_bootstrap)
    return [
        EpsilonGreedy(model, epsilon=settings.epsilon, rng=rng),
        UCB(model, c=settings.ucb_c, temp=settings.softmax_temp, rng=rng),
        ThompsonSampling(ensemble, rng=rng),
    ]


def _plan_to_doors(
    plan: Route | list[Route], doors: list[ProspectContext]
) -> list[ProspectContext]:
    """Flatten a (possibly multi-rep) plan into the doors to service, in visiting order.

    ``plan_route`` builds ``coords = [depot, *doors]``, so node ``k >= 1`` is ``doors[k - 1]``.
    For a team plan the per-rep orders are concatenated (they are disjoint by construction).
    """
    routes = plan if isinstance(plan, list) else [plan]
    seq: list[ProspectContext] = []
    seen: set[int] = set()
    for route in routes:
        for node in route.order:
            if node != 0 and (node - 1) not in seen:
                seen.add(node - 1)
                seq.append(doors[node - 1])
    return seq


def _naive_nn_time(coords: list[tuple[float, float]], tm: np.ndarray, service_s: float) -> float:
    """Total time of a greedy nearest-neighbor tour that visits *every* door (depot at 0)."""
    n = len(coords)
    unvisited = set(range(1, n))
    cur, total = 0, 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda j: tm[cur][j])
        total += float(tm[cur][nxt]) + service_s
        cur = nxt
        unvisited.discard(nxt)
    total += float(tm[cur][0])  # walk back to the depot
    return total


def _subsample_batch(batch: LoggedBatch, max_rows: int, rng: np.random.Generator) -> LoggedBatch:
    """Random subset of a batch (<= max_rows) so OPE stays responsive on large logs."""
    if len(batch) <= max_rows:
        return batch
    idx = rng.choice(len(batch), size=max_rows, replace=False)
    return LoggedBatch(
        contexts=[batch.contexts[i] for i in idx],
        actions=batch.actions[idx],
        rewards=batch.rewards[idx],
        propensities=batch.propensities[idx],
    )


def run_demo(
    *,
    n_logs: int = 20_000,
    shift: int = 60,
    replan_every: int = 10,
    seed: int = 7,
    settings: Settings | None = None,
    ope_max_rows: int = 5_000,
    report_path: Path | None = Path("artifacts/demo_report.json"),
    write: bool = True,
) -> DemoReport:
    """Run the full offline loop for one shift and return a :class:`DemoReport`."""
    settings = (settings or get_settings()).model_copy(update={"seed": seed})

    # 1. Logs → train / held-out split. Relational mode and ``use_simulated_drift`` are
    #    resolved by :func:`generate_logs_for_settings` (flat default stays stationary).
    events, _ = generate_logs_for_settings(n_logs, settings=settings, seed=seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(events))
    n_val = max(50, len(events) // 5)
    ope_events = [events[i] for i in perm[:n_val]]
    train_events = [events[i] for i in perm[n_val:]]

    # 2. Reward model.
    model = RewardModel.fit(train_events, settings=settings)
    if write:
        model.save(settings.model_dir)

    # 3. Policies.
    policies = _build_policies(model, train_events, settings, np.random.default_rng(seed))

    # 4. OPE + gate selection.
    full_batch = LoggedBatch.from_events(ope_events)
    baseline_value = float(full_batch.rewards.mean())  # on-policy value of the logging policy
    ope_batch = _subsample_batch(full_batch, ope_max_rows, np.random.default_rng(seed))
    q_hat = q_matrix(model, ope_batch.contexts)
    gate = PromotionGate(z=settings.ope_z, min_lift=settings.ope_min_lift)

    decisions = {
        p.name: gate.evaluate(p, ope_batch, q_hat, baseline_value=baseline_value) for p in policies
    }
    ope = {
        name: {
            "ips": dec.candidate["ips"].value,
            "snips": dec.candidate["snips"].value,
            "dm": dec.candidate["dm"].value,
            "dr": dec.candidate["dr"].value,
            "dr_lower_bound": dec.lower_bound,
            "promote": float(dec.promote),
        }
        for name, dec in decisions.items()
    }
    selected = max(policies, key=lambda p: decisions[p.name].candidate["dr"].value)
    selected_decision = decisions[selected.name]

    # 5. Simulate the shift over a dense neighborhood, walking the planned route.
    from nba.api.store import EventStore  # local import keeps module import cheap

    contexts = _dense_block(shift, settings, seed + 1)
    # Grading oracle for this shift. In relational mode we rebuild the world over the *repositioned*
    # doors so neighbor edges (and thus social proof) stay consistent with the new geography.
    if settings.dataset_mode == "relational":
        world = rel.world_from_contexts(contexts, settings=settings, seed=seed + 1)
        oracle = oracle_for(settings, world=world)
    else:
        oracle = oracle_for(settings)
    store = EventStore(settings.db_path)
    policy = EthicalPolicy(selected, settings, rng=np.random.default_rng(seed + 2))
    engine = build_distance_engine(settings)
    orch = Orchestrator(
        policy=policy,
        reward_model=model,
        distance_engine=engine,
        store=store,
        settings=settings,
    )

    # Routing stats from the initial plan + a visit-all baseline. A team plan (num_vehicles > 1)
    # returns one route per rep; reps walk in parallel, so wall-clock time is the slowest rep and
    # throughput is the union of doors served.
    initial = orch.plan_route(contexts)
    initial_routes = initial if isinstance(initial, list) else [initial]
    team_visited = sum(len(r.visited) for r in initial_routes)
    team_dropped = len(initial_routes[0].dropped)  # dropped is the global no-rep-served set
    team_route_time_s = max(r.total_time_s for r in initial_routes)

    door_coords = [(c.lat, c.lon) for c in contexts]
    depot = (
        float(np.mean([la for la, _ in door_coords])),
        float(np.mean([lo for _, lo in door_coords])),
    )
    full_coords = [depot, *door_coords]
    tm = engine.time_matrix(full_coords)
    naive_time = _naive_nn_time(full_coords, tm, service_s=120.0)

    sim_rng = np.random.default_rng(seed + 3)
    remaining = list(contexts)
    route = initial
    order = _plan_to_doors(route, remaining)

    regret_curve: list[float] = []
    chosen_true: list[float] = []
    best_true: list[float] = []
    serviced: list[ProspectContext] = []
    cum_regret = 0.0
    realized = 0.0
    step = 0

    while order:
        door = order.pop(0)
        result = orch.recommend(door)
        outcome = oracle.sample_outcome(door, result.action, sim_rng)
        orch.feedback(result.decision_id, outcome)

        realized += REWARD[outcome]
        chosen_value = oracle.true_reward(door, result.action)
        best_value = oracle.true_reward(door, oracle.true_best_action(door))
        cum_regret += best_value - chosen_value
        regret_curve.append(cum_regret)
        chosen_true.append(chosen_value)
        best_true.append(best_value)
        serviced.append(door)
        remaining.remove(door)
        step += 1

        if step % replan_every == 0 and remaining:
            route = orch.replan(remaining)
            order = _plan_to_doors(route, remaining)

    # 6. Baselines on the *same* serviced doors (expected reward, oracle, eval-only).
    bandit_expected = float(sum(chosen_true))
    uniform_expected = float(
        sum(np.mean([oracle.true_reward(c, a) for a in ACTIONS]) for c in serviced)
    )
    exploit_expected = float(sum(oracle.true_reward(c, model.best_action(c)) for c in serviced))
    avg_regret_curve = [r / (i + 1) for i, r in enumerate(regret_curve)]

    stored = store.load_events()
    propensities = [e.propensity for e in stored]

    report = DemoReport(
        seed=seed,
        n_logs=n_logs,
        shift=shift,
        replan_every=replan_every,
        baseline_value=baseline_value,
        selected_policy=selected.name,
        gate_promoted=selected_decision.promote,
        selected_dr_lower_bound=selected_decision.lower_bound,
        ope=ope,
        expected_reward={
            "bandit": bandit_expected,
            "uniform": uniform_expected,
            "exploit": exploit_expected,
            "oracle_best": float(sum(best_true)),
        },
        realized_reward_bandit=realized,
        regret_curve=regret_curve,
        avg_regret_curve=avg_regret_curve,
        routing={
            "candidates": float(len(contexts)),
            "visited": float(team_visited),
            "dropped": float(team_dropped),
            "route_time_min": team_route_time_s / 60.0,
            "naive_visit_all_time_min": naive_time / 60.0,
            "time_saved_min": (naive_time - team_route_time_s) / 60.0,
        },
        n_decisions=len(serviced),
        min_propensity=float(min(propensities)) if propensities else 0.0,
        model=model,
        policy=policy,
        settings=settings,
    )

    if write and report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_json(), indent=2))

    store.close()
    return report


def _print_report(report: DemoReport) -> None:
    print(f"\n{'=' * 70}")
    print(
        f"NBA end-to-end demo  (seed={report.seed}, n_logs={report.n_logs:,}, shift={report.shift})"
    )
    print("=" * 70)

    print(f"\nLogging baseline value (held-out): {report.baseline_value:+.4f}\n")
    print(f"{'policy':<24} {'IPS':>8} {'SNIPS':>8} {'DM':>8} {'DR':>8} {'DR-lb':>8}  gate")
    for name, row in report.ope.items():
        mark = "PROMOTE" if row["promote"] else "hold"
        star = " *" if name == report.selected_policy else "  "
        print(
            f"{name:<24}{star}{row['ips']:>8.4f} {row['snips']:>8.4f} {row['dm']:>8.4f} "
            f"{row['dr']:>8.4f} {row['dr_lower_bound']:>8.4f}  {mark}"
        )
    print(
        f"\nselected: {report.selected_policy}  "
        f"(gate {'PROMOTED' if report.gate_promoted else 'HELD'}, "
        f"DR lb {report.selected_dr_lower_bound:+.4f} vs baseline {report.baseline_value:+.4f})"
    )

    er = report.expected_reward
    print(f"\nExpected reward over {report.n_decisions} serviced doors (oracle, eval-only):")
    print(f"  bandit (selected) : {er['bandit']:+.3f}")
    print(f"  uniform-random    : {er['uniform']:+.3f}")
    print(f"  exploit-only      : {er['exploit']:+.3f}")
    print(f"  bandit beats uniform by {er['bandit'] - er['uniform']:+.3f}")
    print(
        f"  final avg regret/round: {report.avg_regret_curve[-1]:.4f}"
        if report.avg_regret_curve
        else "  (no doors serviced)"
    )

    r = report.routing
    print(
        f"\nRouting: visited {r['visited']:.0f} / dropped {r['dropped']:.0f} "
        f"of {r['candidates']:.0f} candidates"
    )
    print(f"  planned route   : {r['route_time_min']:.1f} min")
    print(f"  visit-all (NN)  : {r['naive_visit_all_time_min']:.1f} min")
    print(f"  walk-time saved : {r['time_saved_min']:.1f} min")

    print(
        f"\nPropensity logged on every decision: "
        f"min p = {report.min_propensity:.4f} (> 0 required for OPE)"
    )
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end NBA demo for one shift.")
    parser.add_argument("--n-logs", type=int, default=20_000)
    parser.add_argument("--shift", type=int, default=60)
    parser.add_argument("--replan-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo_report.json"))
    args = parser.parse_args()

    report = run_demo(
        n_logs=args.n_logs,
        shift=args.shift,
        replan_every=args.replan_every,
        seed=args.seed,
        report_path=args.out,
    )
    _print_report(report)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
