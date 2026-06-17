# Reward Model — Hypotheses

## H1: Isotonic calibration improves DM/DR gate decisions vs raw q

DR leans on q̂; miscalibrated raw scores should bias DM. Hypothesis: gate promote/HOLD flips on some
seeds when using raw vs calibrated q in OPE.

*Evidence: training metrics show calibrated MSE ≤ raw; no gate A/B yet.*

## H2: Recommendation certainty (top calibrated gap) reduces bad field recommendations

When top two actions have similar calibrated value, abstain or explore more. Needs product threshold
tuning on real shift outcomes.

*Evidence: display_calibration notebook analysis only.*

## H3: Bootstrap ensemble size B=16 is sufficient for Thompson; B=4 enough for demo speed

Trade-off between Thompson uncertainty quality and fit time. Demo uses n_bootstrap=2–4; production
default 16.

*Evidence: demo works at B=4; no systematic B sweep.*
