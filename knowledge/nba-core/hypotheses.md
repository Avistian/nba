# NBA Core — Hypotheses

Need more data before promoting to rules.

## H1: Online retraining across shifts will show decreasing per-round regret

A single shift uses a fixed gated policy → stationary regret. Hypothesis: periodic
`RewardModel.fit(load_events())` + re-gating across many shifts produces the textbook downward
regret curve.

*Evidence so far: 0 online-retrain loops implemented. Test when adding continual learning.*

## H2: Parquet batch logs + SQLite serving store remain sufficient until multi-rep concurrency

Current design: parquet for offline batch, SQLite WAL for serving. Hypothesis: no migration needed
until multiple reps write concurrently at high QPS.

*Evidence: single-rep demo works; no load tests yet.*

## H3: `make demo` at default n_logs=20k is too slow for CI (< 5 min target)

Full demo trains models + bootstrap ensemble + OPE; observed ~4+ minutes at 12k logs. Hypothesis: CI
should use a fast fixture path (like `test_e2e` at n_logs=2500) and keep `make demo` as manual
smoke only.

*Evidence: 1 timing observation (12k logs ≈ 4.5 min). Needs 2+ more runs at varied N.*
