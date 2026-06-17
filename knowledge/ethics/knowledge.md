# Ethics — Knowledge

Two layers enforced in code:

## Structural (features)

`ALLOWED_FEATURES` in `data/features.py` — explicit allow-list. Vector built from list, not context
reflection. No `lat`/`lon`/`address_id`/protected attributes in `FEATURE_NAMES`.

Tested: `test_feature_allowlist_excludes_protected_and_geo`,
`test_model_trains_only_on_allowed_features`.

## Behavioral (policy)

`ethics.py`:

- `is_sensitive(ctx)` — flags doors with `prior_interactions >= sensitive_prior_interactions`
  (default 4). Uses behavioral signal, not demographics.
- `cap_exploration(dist, ceiling)` — shrink explore mass toward mode; **preserve full support**.
- `EthicalPolicy` — wraps any `Policy`; caps exploration in sensitive contexts when
  `cap_exploration_in_sensitive=True` (default).

Demo and production shift run through `EthicalPolicy` wrapper.

## Oracle leak prevention

AST scan across learning packages ensures no import/reference to `true_reward`, `latent_scores`,
`true_best_action`, `outcome_probs`.

## Config knobs (`config.py`)

- `cap_exploration_in_sensitive` (default True)
- `sensitive_prior_interactions` (default 4)
- `sensitive_exploration_ceiling` (default 0.05)
