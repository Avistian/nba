"""The orchestration seam: bandit per-door profits feed the TSP-with-Profits router.

The :class:`~nba.pipeline.orchestrator.Orchestrator` is pure Python and dependency-injected
(policy, reward model, distance engine, event store). It is the single place where the "propose"
half (reward model + bandit) meets the "dispose" half (router), and where every decision is logged.
"""
