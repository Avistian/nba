# Phase 2 — D2D simulator + feature substrate

**Depends on:** Phase 1. **Goal:** a self-contained, offline ground-truth world that (a) draws
realistic `ProspectContext`s from Ames/ACS (with a synthetic fallback), (b) defines a latent
conversion model and `true_reward` oracle, and (c) runs a **stochastic logging policy that
records propensity** to emit `data/logs.parquet`. This is what makes the whole pipeline testable
without field data.

## Files to create

```
src/nba/data/ames.py
src/nba/data/simulator.py
scripts/generate_logs.py
tests/test_ames.py
tests/test_simulator.py
```

## `src/nba/data/ames.py`

```python
AMES_URL = "https://raw.githubusercontent.com/.../AmesHousing.csv"  # documented source

def load_ames(settings: Settings) -> pd.DataFrame:
    # cached at data/ames.parquet; if download unavailable → synthetic_ames(n, rng)
def synthetic_ames(n: int, rng: np.random.Generator) -> pd.DataFrame:
    # log-normal property_value, uniform roof age, income correlated w/ value
def map_row_to_context(row, env, rng) -> ProspectContext:
    # property_value ← SalePrice; roof_age ← YearBuilt/RemodAdd; income ~ f(value)+noise
```
- **Offline-first:** any network failure silently falls back to `synthetic_ames`; tests never hit
  the network (mark a network test `slow` and skip if offline).
- Caches parsed frame to `data/ames.parquet` to avoid re-parsing.

## `src/nba/data/simulator.py`

### Environment sampling

```python
def sample_environment(rng) -> dict:   # hour∈[8,20], dow, weather, block_density, neighbor flag
def sample_context(ames_row, rng) -> ProspectContext:   # joins prospect + environment + spatial
```

### Latent ground truth (the world model)

```python
def latent_scores(ctx: ProspectContext, a: Action) -> dict[Outcome, float]:
    """Unnormalized logits per outcome with documented interaction effects:
       - KNOCK_NOW in evening (17–19h) ↑ APPOINTMENT/CLOSED
       - high tenure_years ↑ engagement; high property_value ↑ solar fit
       - neighbor_recent_conversion ↑ social proof
       - PITCH_SECURITY ↑ when block_density low / income high
       - bad weather ↑ SLAMMED/NOT_HOME; SKIP_DOOR → deterministic NOT_HOME-like null
       - prior_interactions high → diminishing returns / ↑ SLAMMED
    """
def outcome_probs(ctx, a, *, rng=None) -> dict[Outcome, float]:   # softmax(latent_scores)
def sample_outcome(ctx, a, rng) -> Outcome:                       # categorical draw
def true_reward(ctx, a) -> float:                                 # Σ p(o)·REWARD[o]  (ORACLE)
def true_best_action(ctx) -> Action:                             # argmax_a true_reward (regret)
```
- `SKIP_DOOR` yields reward 0 deterministically (probability mass on a null outcome) so skipping
  is the genuine baseline.
- **Oracle isolation:** `true_reward`/`true_best_action`/`latent_scores` must never be imported by
  `reward/`, `bandits/`, or `ope/`. Only the simulator and tests/eval use them.

### Logging (behavior) policy — records propensity

```python
def behavior_policy(ctx, rng, *, temp: float = 0.5) -> tuple[Action, float]:
    """Weak heuristic score per action (uses action_cost + a few cheap context cues),
       softmax(score/temp) → full-support distribution; sample action, RETURN (action, p[action]).
       Guarantees p>0 for every arm (positivity/overlap — required for OPE)."""
def action_distribution(ctx, *, temp) -> dict[Action, float]   # for tests / exact propensity
```

### Event generation

```python
def simulate_event(ctx, rng, clock: datetime) -> BanditEvent:   # action+p via behavior, outcome+reward via latent
def generate_logs(n: int, *, settings, seed: int) -> list[BanditEvent]
def logs_to_frame(events) -> pd.DataFrame                        # flat columns for parquet
```

## `scripts/generate_logs.py`

- CLI (`argparse`): `--n 20000 --seed 7 --out data/logs.parquet`.
- Calls `generate_logs`, writes parquet with columns: `ctx.*` (flattened), `action`, `propensity`,
  `reward`, `outcome`, `decision_id`, `timestamp`, `lat`, `lon`.
- Prints summary: arm frequencies, mean reward, min propensity (must be > 0).

## Tests

`tests/test_ames.py`
- `synthetic_ames(100, rng)` has expected columns and positive values; deterministic by seed.
- `load_ames` falls back to synthetic when forced offline (monkeypatch download → raise).

`tests/test_simulator.py`
- **Positivity:** every event `propensity > 0`; `action_distribution` sums to 1 with full support.
- **Reproducibility:** `generate_logs(n, seed=7)` twice → identical frames.
- **Oracle sanity:** for a hand-built "hot" context (evening, high tenure, neighbor converted),
  `true_best_action` is a knock/pitch, not `SKIP_DOOR`; for a "cold/hostile" context, `SKIP_DOOR`
  is competitive.
- **Coverage:** across many sampled contexts each arm is logged > 0 times (overlap holds).
- **Reward range:** sampled rewards ⊆ `REWARD.values()`; `true_reward ∈ [min,max] REWARD`.
- **Oracle isolation (guard test):** import-graph check that `nba.reward`, `nba.bandits`,
  `nba.ope` do not reference `latent_scores`/`true_reward` (grep/AST on module source).

## Acceptance

- `python scripts/generate_logs.py --n 5000 --seed 7` writes a parquet with min propensity > 0
  and all 5 arms represented.
- Logs are reproducible by seed; oracle rankings are intuitively correct on probe contexts.
- No oracle symbol leaks into model/policy/OPE modules.
