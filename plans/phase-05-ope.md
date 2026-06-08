# Phase 5 — Off-policy evaluation + promotion gate

**Depends on:** Phase 4. **Goal:** estimate a candidate policy's value from *logged* data only
(no field deployment) using IPS, DM, and DR; validate our estimators against the **Open Bandit
Pipeline (OBP)**; and gate promotion so a new policy ships only if it beats the logging baseline
within a confidence bound. This is the safety valve before "the bandit proposes" reaches reality.

## Files to create

```
src/nba/ope/__init__.py
src/nba/ope/estimators.py
src/nba/ope/gate.py
scripts/evaluate_policy.py
tests/test_ope.py
```

## `src/nba/ope/estimators.py`

### Data adapter

```python
@dataclass(frozen=True)
class LoggedBatch:
    contexts: list[ProspectContext]
    actions: np.ndarray            # int-encoded over ACTIONS, shape (n,)
    rewards: np.ndarray            # shape (n,)
    propensities: np.ndarray       # logging p, shape (n,), all > 0
    @classmethod
    def from_events(cls, events) -> "LoggedBatch": ...

def eval_action_matrix(policy: Policy, contexts) -> np.ndarray:   # (n, |A|) target probs π_e(a|x)
def q_matrix(model: RewardModel, contexts) -> np.ndarray:        # (n, |A|) q̂(x,a)
```

### Estimators (each returns `OPEResult`)

```python
@dataclass(frozen=True)
class OPEResult:
    estimator: str; value: float; std_err: float; n: int
    def ci(self, z: float) -> tuple[float, float]: ...   # (value - z*se, value + z*se)

def ips(batch, pi_e, *, clip: float | None = None) -> OPEResult:
    # w_i = π_e[i, a_i] / p_i  (optional weight clip for variance control)
    # V = mean(w_i * r_i);  se = std(w_i*r_i)/sqrt(n)
def snips(batch, pi_e, *, clip=None) -> OPEResult:        # self-normalized IPS (lower variance)
    # V = Σ w_i r_i / Σ w_i
def dm(batch, q_hat, pi_e) -> OPEResult:
    # V = mean_i Σ_a π_e[i,a] q_hat[i,a];  se via bootstrap over rows
def dr(batch, q_hat, pi_e, *, clip=None) -> OPEResult:
    # baseline = Σ_a π_e q_hat ;  correction = w_i (r_i − q_hat[i, a_i])
    # V = mean(baseline_i + correction_i);  se = std(per-row)/sqrt(n)
def evaluate_all(batch, pi_e, q_hat, *, clip=None, z=1.96) -> dict[str, OPEResult]
```

- **Weight clipping** (`clip`) caps `w_i` to trade bias for variance; default `None`, expose in CLI.
- **Std errors:** closed-form (per-row variance / √n) for IPS/SNIPS/DR; bootstrap for DM. Provide a
  shared `_bootstrap_se(per_row_values, n_boot, rng)` helper.
- **Numerical guards:** assert all `propensities > 0`; assert `pi_e` rows sum to 1; warn if any
  effective sample size `(Σw)²/Σw²` is tiny (overlap problem).

## `src/nba/ope/gate.py`

```python
@dataclass(frozen=True)
class GateDecision:
    promote: bool
    candidate: dict[str, OPEResult]      # estimator → result for candidate policy
    baseline_value: float
    lift: float                          # candidate DR value − baseline_value
    lower_bound: float                   # candidate DR value − z*se
    reason: str

class PromotionGate:
    def __init__(self, *, z: float, min_lift: float): ...
    def evaluate(self, candidate: Policy, batch: LoggedBatch, q_hat,
                 *, baseline_value: float) -> GateDecision:
        # estimate candidate via DR (primary) + IPS/DM (reported);
        # promote iff (DR.value - z*DR.std_err) > baseline_value + min_lift
```

- **Baseline value** = on-policy value of the logging policy = `mean(rewards)` of the batch
  (the empirical performance that actually happened). A candidate must beat it with margin.
- **Primary estimator = DR** (lower variance than IPS, less biased than DM). IPS/DM reported for
  transparency and disagreement detection (large IPS-vs-DM gap ⇒ distrust, surfaced in `reason`).

## OBP validation (`tests/test_ope.py`)

- Build a small `obp.dataset.SyntheticBanditDataset` (fixed seed); generate logged feedback.
- Define a target policy distribution; compute our `ips/dm/dr` on that batch and OBP's
  `OffPolicyEvaluation` with `[InverseProbabilityWeighting, DirectMethod, DoublyRobust]`.
- **Assert relative error < tol** (e.g. 5%) between our estimates and OBP's for each estimator.
- This pins our math to a reference implementation; mark `slow` if obp import is heavy.

## Other tests

- **Unbiasedness (IPS):** on synthetic data where logging == target, IPS ≈ `mean(reward)`.
- **Variance ordering:** `Var(DR) ≤ Var(IPS)` on the same batch (typical; assert with slack).
- **Gate correctness (oracle-checked via simulator):**
  - a deliberately *worse* target (skip-everything) → `promote == False`.
  - a deliberately *better* target (near-oracle, built from `true_best_action` only inside the
    test) → `promote == True`. Oracle is used only to *construct the test target*, never inside
    `ope/`.
- **Edge:** zero-propensity input raises; `pi_e` not summing to 1 raises; empty batch handled.

## `scripts/evaluate_policy.py`

- CLI: `--logs ... --model artifacts/models --policy {epsilon,ucb,thompson} --clip ... --z 1.96`.
- Builds the batch + target `pi_e`, prints IPS/SNIPS/DM/DR with CIs and the `GateDecision`
  (promote/hold + reason). Exits non-zero if the gate rejects (usable in CI).

## Acceptance

- Our IPS/DM/DR match OBP within tolerance on `SyntheticBanditDataset`.
- DR variance ≤ IPS variance on shared batches; estimators reject zero-overlap input.
- Gate promotes a better-than-logging policy and rejects a worse one (oracle-validated in tests).
- No oracle symbol imported inside `ope/`. `ruff`/`pyright` clean; `pytest` green.
