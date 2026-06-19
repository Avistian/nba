# NBA Core — Hypotheses

Need more data before promoting to rules.

## H1: Conditional retrain on drift will show decreasing per-round regret post-trigger

A single shift uses a fixed gated policy → stationary regret. Hypothesis: **drift-triggered**
retrain (Phase 18: monitor → trigger → DR gate → promote) on accumulated logs produces recovery
toward pre-drift regret after non-stationarity, without the noise of daily blind retrain.

*Evidence so far: Phase 18 planned (`plans/phase-18-drift-monitoring-retrain-loop.md`); 0 monitor/retrain
code yet. Test via `simulate_drift_demo.py` when implemented.*

## H2: Parquet batch logs + SQLite serving store remain sufficient until multi-rep concurrency

Current design: parquet for offline batch, SQLite WAL for serving. Hypothesis: no migration needed
until multiple reps write concurrently at high QPS.

*Evidence: single-rep demo works; no load tests yet.*

## H3: `make demo` at default n_logs=20k is too slow for CI (< 5 min target)

Full demo trains models + bootstrap ensemble + OPE; observed ~4+ minutes at 12k logs. Hypothesis: CI
should use a fast fixture path (like `test_e2e` at n_logs=2500) and keep `make demo` as manual
smoke only.

*Evidence: 1 timing observation (12k logs ≈ 4.5 min). Needs 2+ more runs at varied N.*
