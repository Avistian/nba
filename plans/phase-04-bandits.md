# Phase 4 — Bandit policies (pluggable)

**Depends on:** Phase 3. **Goal:** three exploration strategies — ε-greedy, UCB, Thompson — behind
a single `Policy` protocol. Each returns `(action, propensity)` for logging **and** a full
`action_dist` (probabilities summing to 1, full support) that OPE consumes. "The bandit proposes."

## Files to create

```
src/nba/bandits/__init__.py
src/nba/bandits/base.py
src/nba/bandits/epsilon_greedy.py
src/nba/bandits/ucb.py
src/nba/bandits/thompson.py
tests/test_bandits.py
```

## `src/nba/bandits/base.py`

```python
@runtime_checkable
class Policy(Protocol):
    name: str
    def recommend(self, ctx: ProspectContext, actions=ACTIONS) -> tuple[Action, float]: ...
    def action_dist(self, ctx: ProspectContext, actions=ACTIONS) -> dict[Action, float]: ...

def validate_dist(dist: dict[Action, float], *, full_support=True, tol=1e-9) -> None:
    # sums to 1±tol; all values ≥ 0; if full_support, all > 0  (raises ValueError otherwise)

def sample_from_dist(dist, rng) -> tuple[Action, float]:   # categorical draw → (action, dist[action])

def softmax(scores: np.ndarray, temp: float) -> np.ndarray: # numerically stable
```
- **Why `action_dist` must have full support:** IPS/DR divide by the logging propensity and weight
  by the *target* policy's probability; a zero there is fine for the target but a zero in the
  *logging* policy breaks overlap. To keep every shipped policy safely usable as a logging policy
  too, all three floor their distributions to full support.

## `src/nba/bandits/epsilon_greedy.py`

```python
class EpsilonGreedy:
    name = "epsilon_greedy"
    def __init__(self, model: RewardModel, *, epsilon: float, rng): ...
    def action_dist(self, ctx, actions=ACTIONS):
        # best = argmax model.q_all(ctx); p[a] = eps/|A|; p[best] += (1-eps)
    def recommend(self, ctx, actions=ACTIONS):
        # sample_from_dist(action_dist(ctx), rng)
```
- Edge: ties in argmax → split the exploit mass uniformly among tied arms (deterministic by index
  if `rng` not desired). Document the tie rule.

## `src/nba/bandits/ucb.py`

```python
class UCB:
    name = "ucb"
    def __init__(self, model: RewardModel, *, c: float, temp: float, rng,
                 bucketizer: Callable[[ProspectContext], Hashable] | None = None): ...
    # per (context-bucket, arm) visit counts in a dict; t = total pulls in bucket
    def _bonus(self, bucket, a) -> float:        # c * sqrt(ln(t+1) / (n[bucket,a]+1))
    def action_dist(self, ctx, actions=ACTIONS):
        # scores = q_all(ctx) + bonus per arm; dist = softmax(scores, temp)  (smooth, full support)
    def recommend(self, ctx, actions=ACTIONS):
        # draw from action_dist; INCREMENT count for the drawn (bucket, arm)
    def update(self, ctx, a):                    # explicit count bump (used by online loop)
```
- **Counts need a context bucket** (continuous contexts never repeat). Default bucketizer:
  coarse-bin a few salient features (e.g. `hour∈{morning,afternoon,evening}` × `value-tercile`).
  Document that this is a pragmatic discretization, not LinUCB; note LinUCB as a future swap.
- Softmax-of-UCB-scores gives a differentiable, full-support `action_dist` for OPE (a hard argmax
  UCB would be degenerate); `temp` from config controls exploration sharpness.

## `src/nba/bandits/thompson.py`

```python
class BootstrapEnsemble:
    """B reward models, each fit on a bootstrap resample of the logs (seed offset per member)."""
    @classmethod
    def fit(cls, events, *, settings, n_models: int) -> "BootstrapEnsemble": ...
    def q_all_members(self, ctx) -> np.ndarray:   # shape (B, |A|)

class ThompsonSampling:
    name = "thompson"
    def __init__(self, ensemble: BootstrapEnsemble, *, rng): ...
    def recommend(self, ctx, actions=ACTIONS):
        # pick member m ~ Uniform(B); a = argmax member_m.q_all(ctx); p = action_dist[a]
    def action_dist(self, ctx, actions=ACTIONS):
        # Monte-Carlo: fraction of members whose argmax is each arm; floor to full support (+ε, renorm)
```
- Posterior-over-`q` approximated by bootstrap ensemble (reuses Phase 3 model; chosen default in
  PLAN.md). `action_dist` is the MC estimate of P(arm is best); ε-floor keeps overlap.
- `n_models = settings.n_bootstrap`. Fitting B LightGBMs is the cost; cache the ensemble to disk
  (joblib) like `RewardModel`.

## Tests

`tests/test_bandits.py` (parametrized over all three policies)
- **Protocol:** `isinstance(policy, Policy)` true; `name` set.
- **Distribution contract:** `action_dist` sums to 1±1e-9, all values > 0 (`validate_dist`).
- **Propensity match:** for a fixed rng, `recommend` returns `(a, p)` with
  `p == action_dist(ctx)[a]`.
- **Determinism:** same seed → same action sequence over N calls.
- **Greedy limits:** `EpsilonGreedy(eps=0)` → near-degenerate (argmax mass 1−0); `eps=1` →
  uniform `1/|A|`.
- **UCB exploration:** an unpulled arm gets a higher bonus; after many `update`s its bonus shrinks.
- **Thompson:** with a strongly dominant arm in a synthetic model, its `action_dist` mass → high;
  ε-floor keeps all arms > 0.
- **Smoke beats uniform:** in a closed online loop against the simulator, mean reward of each
  policy > uniform-random baseline (seeded, modest N) — sanity, asserted with margin.

## Acceptance

- All three satisfy `Policy`; `action_dist` is a valid full-support distribution; `recommend`
  propensity equals the chosen arm's probability.
- Policies are deterministic under a fixed rng and pluggable by construction (swap one line).
- `ruff`/`pyright` clean; `pytest` green.
