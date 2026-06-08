# Phase 1 — Schema + reward function + features

**Depends on:** Phase 0. **Goal:** the domain vocabulary every other module imports — actions,
outcomes, the reward map, the `ProspectContext`, the logged `BanditEvent`, and a deterministic
`featurize()` that turns `(context, action)` into a fixed-width vector through an ethics
allow-list.

## Files to create

```
src/nba/schema.py
src/nba/data/__init__.py
src/nba/data/features.py
tests/test_schema.py
tests/test_features.py
```

## `src/nba/schema.py`

### Enums

```python
class Action(str, Enum):
    KNOCK_NOW     = "knock_now"
    LEAVE_FLYER   = "leave_flyer"
    SKIP_DOOR     = "skip_door"
    PITCH_SOLAR   = "pitch_solar"
    PITCH_SECURITY= "pitch_security"

ACTIONS: tuple[Action, ...] = tuple(Action)   # canonical order for one-hot/q_all

class Outcome(str, Enum):
    SLAMMED     = "slammed"      # hostile / negative
    NOT_HOME    = "not_home"
    INFO        = "info_given"   # micro-conversion
    APPOINTMENT = "appointment"  # strong intent
    CLOSED      = "closed"       # sale
```

### Reward map (monotone, documented)

```python
REWARD: dict[Outcome, float] = {
    Outcome.SLAMMED: -0.2, Outcome.NOT_HOME: 0.0, Outcome.INFO: 0.1,
    Outcome.APPOINTMENT: 0.3, Outcome.CLOSED: 1.0,
}
def reward_for(outcome: Outcome) -> float: return REWARD[outcome]
```
- Note in docstring: `SLAMMED` is negative to encode reputational cost of bad knocks (this makes
  `SKIP_DOOR` a real opportunity-cost decision, not a free option).

### Action cost (opportunity/effort prior, used by router profit and as a weak heuristic)

```python
ACTION_COST: dict[Action, float] = {
    Action.SKIP_DOOR: 0.0, Action.LEAVE_FLYER: 0.02,
    Action.PITCH_SOLAR: 0.05, Action.PITCH_SECURITY: 0.05, Action.KNOCK_NOW: 0.05,
}
def action_cost(a: Action) -> float: ...
```

### Context (pydantic v2 `BaseModel`, `frozen=True`, `extra="forbid"`)

```python
class ProspectContext(BaseModel):
    # identity / geo
    address_id: str
    lat: float; lon: float
    # prospect block
    property_value: float          # USD
    roof_age_years: float
    est_income: float              # USD/yr
    tenure_years: float            # how long resident at address
    prior_interactions: int        # times contacted before
    # environment block
    hour: int                      # 0..23
    dow: int                       # 0..6
    weather: Literal["clear","rain","cold","hot"]
    block_density: float           # doors per block
    neighbor_recent_conversion: bool
    # spatial block
    distance_from_rep_km: float
    nearby_high_reward_density: float  # local density of promising doors
```
- Field validators: `0 <= hour <= 23`, `0 <= dow <= 6`, non-negative magnitudes.
- **No** protected attributes (race, religion, etc.) — enforced structurally (they simply don't
  exist on the model) and again by the features allow-list.

### Logged event

```python
class BanditEvent(BaseModel):
    context: ProspectContext
    action: Action
    propensity: float = Field(gt=0.0, le=1.0)   # p(action | context) under logging policy
    reward: float | None = None                 # filled at feedback time
    outcome: Outcome | None = None
    timestamp: datetime
    decision_id: str                            # uuid4; links decision↔outcome
```
- `propensity` is **required and > 0** at construction — this is the contract that makes OPE
  possible. Reward/outcome are nullable until feedback arrives.

## `src/nba/data/features.py`

```python
ALLOWED_FEATURES: tuple[str, ...] = (
    "property_value", "roof_age_years", "est_income", "tenure_years",
    "prior_interactions", "hour", "dow", "block_density",
    "neighbor_recent_conversion", "distance_from_rep_km", "nearby_high_reward_density",
)  # weather is one-hot expanded separately; geo/address never enter the model

WEATHER_LEVELS = ("clear", "rain", "cold", "hot")

def context_vector(ctx: ProspectContext) -> np.ndarray: ...   # numeric block + weather one-hot
def action_onehot(a: Action) -> np.ndarray: ...               # len == len(ACTIONS)
def featurize(ctx: ProspectContext, a: Action) -> np.ndarray: # concat, fixed dtype float64
def featurize_batch(ctx: ProspectContext, actions=ACTIONS) -> np.ndarray:  # (|A|, d) for q_all
FEATURE_NAMES: list[str]                                      # len == n_features()
def n_features() -> int: ...
```

- **Column order is frozen**: `[ALLOWED numeric…] + [weather one-hot…] + [action one-hot…]`.
  `FEATURE_NAMES` is the single source of truth; models store/verify it.
- `featurize` must reject (assert) any attempt to read a field outside `ALLOWED_FEATURES` — guard
  by constructing the vector only from the allow-list, not by reflecting over the model.
- Booleans → `{0.0, 1.0}`. No scaling here (LightGBM is scale-invariant); document that choice.

## Tests

`tests/test_schema.py`
- `BanditEvent(propensity=0)` and `propensity=1.5` raise `ValidationError`.
- `REWARD` strictly ordered: `SLAMMED < NOT_HOME < INFO < APPOINTMENT < CLOSED`.
- `ProspectContext(extra=...)` rejects unknown fields; `hour=24` rejected.
- `ACTIONS` order stable and length 5.

`tests/test_features.py`
- `len(featurize(ctx, a)) == n_features() == len(FEATURE_NAMES)` for all actions.
- Determinism: same input → byte-identical array; different action → different one-hot block only.
- `featurize_batch(ctx)` shape `(5, n_features())`; row `i` equals `featurize(ctx, ACTIONS[i])`.
- Weather one-hot sums to 1; unknown weather impossible (enum/Literal guarded upstream).
- Allow-list guard: assert no geo/address columns appear in `FEATURE_NAMES`.

## Acceptance

- Every `BanditEvent` carries a valid `propensity ∈ (0,1]` and a `decision_id`.
- `featurize` is deterministic with stable width and column order across the whole codebase.
- `FEATURE_NAMES` excludes lat/lon/address_id and any protected attribute.
- `ruff`/`pyright` clean; `pytest` green.
