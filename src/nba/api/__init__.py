"""The HTTP edge: an append-only event store and a thin FastAPI service over the orchestrator.

The API layer only adapts HTTP <-> :class:`~nba.pipeline.orchestrator.Orchestrator` methods; all
decision logic lives in the pipeline. Every ``/recommend`` persists its propensity to the
:class:`~nba.api.store.EventStore`, so the logs can later feed the OPE gate (Phase 5).
"""
