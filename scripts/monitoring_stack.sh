#!/usr/bin/env bash
# Bring the optional Grafana + Prometheus dev stack up (or down).
#
# This is a thin wrapper around `docker compose -f monitoring/docker-compose.monitoring.yml`.
# It is OFF by default (no Docker daemon required for tests or CI). Requires `docker`
# and `docker compose` on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/monitoring/docker-compose.monitoring.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is not installed or not on PATH" >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "error: 'docker compose' plugin is unavailable" >&2
  exit 1
fi

ACTION="${1:-up}"
case "${ACTION}" in
  up)
    echo "Starting NBA monitoring stack (Grafana :3000, Prometheus :9090)..."
    docker compose -f "${COMPOSE_FILE}" up -d
    echo
    echo "Grafana     -> http://localhost:3000  (admin/admin, change on first login)"
    echo "Prometheus  -> http://localhost:9090"
    echo "Dashboard   -> 'NBA Ops' (provisioned automatically)"
    echo
    echo "Start the metrics exporter on the host:"
    echo "  NBA_METRICS_EXPORTER_ENABLED=1 uv run python scripts/run_metrics_exporter.py"
    ;;
  down)
    echo "Stopping NBA monitoring stack..."
    docker compose -f "${COMPOSE_FILE}" down
    ;;
  *)
    echo "usage: $0 {up|down}" >&2
    exit 2
    ;;
esac
