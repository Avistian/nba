# Phase 17 — Experiment leaderboard (lift/regression evaluation for every phase)

**Depends on:** Phase 9 (relational dataset), Phase 5 (OPE estimators + gate), Phase 8 (demo
metrics). **Built right after the relational dataset (Phase 9) and before any upgrade (Phases 10-16)
is evaluated** — it is sequenced after Phase 9 so experiments can be graded on **both** the flat and
the relational datasets, and so every later upgrade has a leaderboard to prove its value against.
(Like Phase 6 — "parallel with 3-5" — its position in the build order is set by its dependencies, not
by its file number.) **Goal:** a single, **append-only, logged leaderboard** of experiments so each
feature flag (each phase) is measured against the baseline and explicitly judged a **lift, a
regression, or neutral**. No upgrade is "done" until it has a leaderboard row that clears the same DR
gate; a row that regresses the baseline **blocks** that upgrade's adoption. This operationalizes the
doc 11 §10 yardsticks ("every upgrade must prove itself"). Step-by-step build in
[docs/21](../docs/21-experiment-leaderboard.md).

## Build sequence (where this slots in)

```
Phase 9 (relational dataset)  ->  Phase 17 (this leaderboard)  ->  Phases 10-16 (each upgrade,
                                                                    tested + proven on the board)
```

The relational dataset comes first because the leaderboard must be able to grade experiments on it;
the leaderboard comes next because every upgrade after it must record a lift/neutral row (no
regressions) before it is adopted.

> **Why a leaderboard.** Today a candidate is judged ad-hoc by `scripts/run_demo.py` /
> `scripts/evaluate_policy.py`. With eight new flags that **compose**, we need a durable, comparable
> record: which flag combination, on which dataset/seed, produced which metrics, and whether it beat
> the baseline through the gate. Mirrors the repo's append-only event-store ethos: results are facts,
> never overwritten.

## Feature flags (added to `src/nba/config.py` `Settings`)

| Flag (`NBA_*`) | Type / default | Effect |
|---|---|---|
| `leaderboard_path` | `Path = Path("artifacts/leaderboard.jsonl")` | Append-only experiment log. |
| `baseline_experiment_id` | `str = "baseline"` | The reference run (all upgrade flags off = today's pipeline) every experiment is compared against. |
| `eval_n_shifts` | `int = 50` | Simulated shifts per experiment (variance/CVaR need repeats). |
| `eval_seeds` | `tuple[int, ...] = (7,)` | Seeds swept per experiment for reproducible spread. |

These are infra knobs; they do not alter the served loop.

## Files to create

```
src/nba/eval/__init__.py
src/nba/eval/metrics.py        # the metric set computed per experiment
src/nba/eval/leaderboard.py    # ExperimentRecord + append-only store + ranking + verdict
scripts/run_experiment.py      # run one named flag-config, score it, append a row, print the board
tests/test_leaderboard.py
tests/test_eval_metrics.py
```

## `src/nba/eval/metrics.py`

The common, comparable metric set (the doc 11 §10 yardsticks), computed from a set of simulated shifts
using the simulator oracle **for grading only** (never for serving):

```python
@dataclass(frozen=True)
class ExperimentMetrics:
    realized_shift_value_mean: float    # PRIMARY: total true reward captured / shift
    realized_shift_value_std: float     # variance of the above across shifts/seeds
    realized_shift_value_cvar: float    # mean of the worst eval_cvar_alpha tail (downside)
    decision_regret_mean: float         # value lost vs the oracle that knew true prizes (U2)
    ope_value: float                    # DR point estimate of the policy value
    ope_lcb: float                      # DR lower confidence bound (the gate quantity)
    optimality_gap: float | None        # learned-router value / OR-Tools value (U4 only)
    route_time_s_mean: float            # operational cost (sanity)

def evaluate(orchestrator, *, settings, n_shifts, seeds) -> ExperimentMetrics:
    """Walk n_shifts simulated shifts under the orchestrator's current flag config; aggregate the
    metrics above. Reuses the run_demo machinery so a leaderboard run == a graded demo run."""
```

- **Oracle hygiene:** `metrics.py` may import the simulator oracle (it is eval code, like the demo and
  tests), but it lives under `nba.eval`, which the AST guard already excludes from the *serving*
  modules. It never feeds the oracle into a model.

## `src/nba/eval/leaderboard.py`

```python
@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str                  # human label, e.g. "phase11-risk-kappa0.5"
    phase: str                          # "09".."16" (or "baseline")
    dataset_mode: str                   # "flat" | "relational"
    flags: dict[str, object]            # the NBA_* config snapshot that defines the experiment
    seeds: list[int]
    metrics: ExperimentMetrics
    baseline_id: str
    deltas: dict[str, float]            # metric -> (this - baseline); sign normalized so + == better
    gate_passed: bool                   # DR LCB beats baseline + min_lift (reuses ope/gate.py)
    verdict: Literal["lift", "regression", "neutral"]
    git_rev: str | None
    timestamp: datetime

def record_experiment(metrics, *, settings, experiment_id, phase, flags, baseline) -> ExperimentRecord:
    """Compute deltas vs `baseline`, run the existing PromotionGate on the DR LCB, derive the verdict
    (lift iff primary metric improves AND gate_passed; regression iff primary metric drops materially;
    else neutral), APPEND one JSONL line. Never mutates prior rows."""

def load_leaderboard(path) -> list[ExperimentRecord]
def baseline_record(records, baseline_id) -> ExperimentRecord | None
def rank(records, *, metric="realized_shift_value_mean") -> list[ExperimentRecord]
def render_table(records) -> str        # markdown leaderboard, best-first, with verdict + delta + gate
```

- **Verdict rule (the core ask):** an experiment is a **lift** only if the **primary** metric
  (`realized_shift_value_mean`) rises **and** `gate_passed` (the DR lower bound clears baseline +
  `ope_min_lift`) — i.e. the improvement is real, not noise. A material drop in the primary metric is
  a **regression**; everything else is **neutral**. The threshold for "material" reuses
  `Settings.ope_min_lift`.
- **Append-only:** writes one JSON line per run; corrections are new rows; readers take the latest by
  timestamp/id. Same discipline as `api/store.py`.

## `scripts/run_experiment.py`

- CLI: `--experiment-id phase11-risk-kappa0.5 --phase 11 --set NBA_USE_RISK_AWARE_ROUTING=1 NBA_RISK_KAPPA=0.5 [--dataset relational] [--baseline baseline]`.
- Builds an orchestrator under the requested flags, runs `eval.evaluate`, calls `record_experiment`,
  then prints `render_table(load_leaderboard())` so the updated board (with this run's lift/regression
  verdict and delta vs baseline) is visible immediately.
- A `--baseline-only` mode records the all-flags-off reference row once.
- Optionally writes/refreshes a human-readable `artifacts/leaderboard.md` snapshot.

## Tests

`tests/test_eval_metrics.py`
- `evaluate` is deterministic for fixed seeds; metrics are within sane bounds (value in REWARD range,
  std ≥ 0, regret ≥ 0).
- Uses the oracle only for grading (no oracle symbol reaches a model — the AST guard still passes).

`tests/test_leaderboard.py`
- **Append-only:** two `record_experiment` calls produce two lines; the first is never modified.
- **Verdict logic:** a synthetic experiment that beats baseline on the primary metric *and* passes the
  gate => `"lift"`; one that beats the mean but fails the gate => `"neutral"`; one that drops the
  primary metric beyond `ope_min_lift` => `"regression"`.
- **Deltas sign-normalized:** lower-is-better metrics (regret, variance) have their delta sign flipped
  so `+` always means "better."
- `render_table` ranks best-first and is stable/reproducible.

## Acceptance — ✅ built

- [x] `python scripts/run_experiment.py --baseline-only` writes the reference row; a subsequent
  `run_experiment` for any phase appends a row tagged **lift / regression / neutral** vs that baseline,
  with per-metric deltas and the DR gate result. The shipped `artifacts/leaderboard.jsonl` (+ `.md`)
  carries `baseline` and `phase09-relational` (a deliberate **neutral**).
- [x] The leaderboard is append-only, reproducible by seed, and uses the oracle for grading only
  (`tests/test_leaderboard.py`, `tests/test_eval_metrics.py`).
- [x] `ruff` / `pyright` clean; `pytest` green.
- Built via `src/nba/eval/{oracle,metrics,leaderboard}.py` and `scripts/run_experiment.py`. Grading
  indirection decision:
  [decisions/2026-06-18-dataset-aware-grading-oracle.md](../decisions/2026-06-18-dataset-aware-grading-oracle.md).
