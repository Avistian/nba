# Bandits & OPE — Hypotheses

## H1: Reward-scaled UCB (c≈0.3, temp≈0.1) beats default knobs after tuning

Default UCB collapses toward uniform because optimism bonus dwarfs O(0.1) q-gaps.

*Evidence: 1 architectural analysis + demo where UCB underperforms Thompson/ε-greedy. Needs
systematic grid search logged as decision.*

## H2: DR lower-bound gate reduces bad promotions vs point-estimate gate

Conservative gate causes more HOLDs but fewer false promotes. Need A/B on simulated policy
rotations to quantify false promote rate.

*Evidence: demo shows HOLD at 12k logs despite competitive DR point estimate. Need 3+ seeds.*

## H3: OPE subsampling (ope_max_rows) does not bias gate materially at 400–2000 rows

`run_demo` subsamples held-out batch for speed. Hypothesis: promotion decision stable vs full batch
when ESS is healthy.

*Evidence: 0 systematic subsample-vs-full comparison yet.*
