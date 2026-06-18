"""Evaluation/grading utilities (experiment leaderboard, metrics, oracle resolver).

This package is *eval code*: like the demo and the tests, it may use the simulator oracle for
**grading only** (never for serving). The repo's oracle-leak AST guard intentionally excludes
``nba.eval`` from the serving packages it scans.
"""
