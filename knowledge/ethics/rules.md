# Ethics — Rules

## R1: Never add protected or geo/identity fields to ALLOWED_FEATURES

Redlining and privacy risk. If spatial signal needed, use aggregate non-identifying fields already
allowed (e.g. `block_density`, `distance_from_rep_km`) — not raw lat/lon in the model.

*Confirmed: Phase 1 design, Phase 8 ethics tests.*

## R2: Sensitive-context cap must preserve full support

`cap_exploration` keeps every arm p > 0 so IPS/DR remain valid on sensitive-door logs.

*Confirmed: `test_cap_exploration_preserves_support_and_caps_mass`,
`test_ethical_policy_caps_exploration_in_sensitive_context`.*

## R3: Run repo-wide oracle-leak test when adding learning-module files

Extend `_LEARNING_PACKAGES` list if new packages added. Any new `.py` under those paths gets AST
scanned.

*Confirmed: `test_no_oracle_leak` parametrized over all learning files.*

## R4: Wrap production/demo policies with EthicalPolicy when exploration enabled

Don't bypass the cap for convenience in demos — tests and demo should reflect production guardrails.

*Confirmed: `run_demo.py` uses `EthicalPolicy`.*
