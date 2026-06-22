"""Long-lived HTTP exporter of NBA drift metrics in Prometheus text format.

When ``NBA_METRICS_EXPORTER_ENABLED=0`` (default), ``--once`` is a clean no-op
exit 0. When enabled, ``--once`` dumps the metrics text to stdout; without
``--once`` the script serves ``/metrics`` on ``metrics_exporter_port``.

The exporter reads artifacts (``drift_reports.jsonl``, ``retrain_audit.jsonl``,
``deployed.json``) every ``metrics_refresh_seconds`` — there is no in-process
state and no new database. It is the only ``monitoring`` process that ever
opens a socket; ``simulate_drift_demo.py`` and the unit tests use ``--once``.

Usage:
    NBA_METRICS_EXPORTER_ENABLED=1 uv run python scripts/run_metrics_exporter.py
    uv run python scripts/run_metrics_exporter.py --once
"""

from __future__ import annotations

import argparse
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Make ``nba`` importable when run via ``uv run python scripts/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba.config import Settings  # noqa: E402
from nba.monitoring.exporter import build_snapshot, render_prometheus_text  # noqa: E402


def _open_store(settings: Settings):
    """Open the EventStore when ``settings.db_path`` exists."""
    from nba.api.store import EventStore  # noqa: PLC0415

    if not settings.db_path.exists():
        return None
    return EventStore(settings.db_path)


def _render(settings: Settings) -> str:
    store = _open_store(settings)
    try:
        snapshot = build_snapshot(settings=settings, store=store)
    finally:
        if store is not None:
            store.close()
    return render_prometheus_text(snapshot, settings=settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Prometheus drift-metrics exporter")
    parser.add_argument(
        "--port", type=int, default=None, help="HTTP port (default: settings.metrics_exporter_port)"
    )
    parser.add_argument(
        "--once", action="store_true", help="Print metrics once to stdout and exit (no HTTP server)"
    )
    args = parser.parse_args()

    settings = Settings()
    port = args.port if args.port is not None else settings.metrics_exporter_port

    if not settings.metrics_exporter_enabled and args.once:
        print(
            "metrics_exporter_enabled=False; nothing to emit (set NBA_METRICS_EXPORTER_ENABLED=1)."
        )
        return
    if not settings.metrics_exporter_enabled:
        print(
            "metrics_exporter_enabled=False; refusing to start the HTTP server. "
            "Set NBA_METRICS_EXPORTER_ENABLED=1 or use --once."
        )
        return

    if args.once:
        sys.stdout.write(_render(settings))
        return

    # Long-lived HTTP server on the configured port.
    handler_cls = _make_handler(settings)

    server = ThreadingHTTPServer(("0.0.0.0", port), handler_cls)
    print(
        "nba metrics exporter listening on "
        f":{port}/metrics (refresh={settings.metrics_refresh_seconds}s)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def _make_handler(settings: Settings) -> type[BaseHTTPRequestHandler]:
    """Build a request handler that re-reads artifacts every refresh window."""

    class _MetricsHandler(BaseHTTPRequestHandler):
        _last_refresh: float = 0.0
        _cached: str = ""

        def do_GET(self) -> None:  # noqa: N802 - http.server requires this name
            if self.path != "/metrics" and self.path != "/":
                self.send_response(404)
                self.end_headers()
                return
            now = time.monotonic()
            if not self._cached or (now - self._last_refresh) >= settings.metrics_refresh_seconds:
                self._cached = _render(settings)
                _MetricsHandler._last_refresh = now
            body = self._cached.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - signature
            # Silence the default stderr access log; Prometheus scrapes are noisy.
            return

    return _MetricsHandler


if __name__ == "__main__":
    main()
