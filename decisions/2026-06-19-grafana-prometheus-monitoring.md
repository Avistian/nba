## Decision: Grafana + Prometheus for Phase 18 ops dashboard (optional, off by default)

## Context: Phase 18 needs an on-call view of drift signals, overlap health, and retrain audit. The user asked for something like a Grafana dashboard in the phase plan.

## Alternatives considered:
- **Grafana + Prometheus** — industry-standard metrics stack; version-controlled dashboard JSON; local Docker Compose.
- **Grafana + SQLite datasource only** — fewer moving parts, but weak time-series story and community plugin dependency.
- **FastAPI `/monitoring` HTML page** — no Docker, but bespoke UI that diverges from production ops norms.
- **Notebook-only plots** — already planned; insufficient for live "watch the demo" ops feel.

## Reasoning: Prometheus pull from a thin exporter reading existing JSONL/SQLite artifacts preserves append-only source of truth, stays testable without Docker (`--once` metrics dump), and matches how production ML ops teams visualize model health. Grafana dashboard JSON is provisioned and reviewable in PRs.

## Trade-offs accepted:
- Docker required only when `use_monitoring_dashboard=True` (dev/demo); CI stays Docker-free.
- Exporter adds a small read path over artifacts; must not block or mutate the serve loop.
- Alert contact points are documented but not committed (no Slack webhooks in repo).

## Supersedes: (none)
