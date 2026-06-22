"""Phase 18 — drift monitoring + conditional retraining loop.

Drift signals (``signals.py``), trigger evaluation (``triggers.py``), the
conditional retrain loop (``retrain.py``), a Prometheus text exporter
(``exporter.py``), and JSONL/SQLite readers (``store_reader.py``).

This package **never** imports the simulator oracle (``nba.data.simulator``,
``nba.data.relational_simulator``, ``nba.data.drift``). The Phase 2 AST guard
in ``tests/test_ethics.py`` scans every file here.
"""
