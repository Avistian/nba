# Quality Criteria — NBA Project

Evaluate every task against these checks before marking complete. Update `Last triggered` when a
criterion catches a real issue.

**Please review** — initial criteria seeded from Phase 8 completion standards. Adjust severity
and add categories as the project evolves.

---

## Category: Toolchain & CI

**Severity:** blocking

**Criteria:**

- [ ] `uv run ruff check .` passes with no errors
- [ ] `uv run ruff format --check .` passes (or format applied)
- [ ] `uv run pyright` reports 0 errors on `src/`, `tests/`, `scripts/`
- [ ] `uv run pytest` passes (note count; 1 skipped OBP slow test is expected)
- [ ] New/changed code has corresponding tests when behavior changes

**Source:** Makefile `check` target, project convention

**Last triggered:** 2026-06-12 (Phase 8 completion)

---

## Category: Oracle isolation

**Severity:** blocking

**Criteria:**

- [ ] No new imports of `true_reward`, `latent_scores`, `true_best_action`, `outcome_probs` in
      learning packages (`reward`, `bandits`, `ope`, `routing`, `api`, `pipeline`)
- [ ] `tests/test_ethics.py::test_no_oracle_leak` still passes after adding learning-module files

**Source:** ARCHITECTURE.md §2, decisions/2026-06-01-oracle-isolation.md

**Last triggered:** 2026-06-18 (Phase 9 — guard prefix `nba.data.sim` would have missed
`relational_simulator`; extended to also match it so the relational oracle stays isolated)

---

## Category: Dataset contract & backward compatibility

**Severity:** blocking

**Criteria:**

- [ ] A new dataset emits a **schema-identical** `BanditEvent` stream (no new fields on
      `BanditEvent` / `ProspectContext`; `extra="forbid"` preserved)
- [ ] Extra structure rides in additive sidecars / optional non-model DataFrame columns that
      `frame_to_events` ignores; round-trip stays identical
- [ ] `dataset_mode="flat"` (default) path is byte-identical — verified by a determinism regression
      (`tests/test_demo_dataset_modes.py`); no new heavy deps imported on the default path
- [ ] A degenerate new world reproduces the flat oracle within tolerance
- [ ] Graph node/edge features stay on the allow-list mirroring `features.ALLOWED_FEATURES`

**Source:** knowledge/dataset-eval/rules.md R1–R4, decisions/2026-06-18-relational-dataset-contract.md

**Last triggered:** 2026-06-18 (Phase 9 relational dataset)

---

## Category: Experiment leaderboard

**Severity:** warning

**Criteria:**

- [ ] `leaderboard.jsonl` stays append-only (corrections are new rows)
- [ ] A **lift** requires both a primary-metric gain and clearing the DR gate; deltas sign-normalized
- [ ] A new dataset substrate (model/router unchanged) is graded **neutral / non-regression**, not a lift
- [ ] Grading reaches the oracle only via `eval/oracle.py` (`oracle_for`), never a learner import

**Source:** knowledge/dataset-eval/rules.md R5–R6, decisions/2026-06-18-dataset-aware-grading-oracle.md

**Last triggered:** 2026-06-18 (Phase 17 leaderboard)

---

## Category: Propensity & overlap

**Severity:** blocking

**Criteria:**

- [ ] Every `recommend` path logs `propensity > 0`
- [ ] Policy `action_dist` has full support (all arms > 0); `validate_dist` used
- [ ] `LoggedBatch` / store tests still pass for overlap invariants

**Source:** knowledge/bandits-ope/rules.md R1–R3

**Last triggered:** never

---

## Category: Ethics & features

**Severity:** blocking

**Criteria:**

- [ ] No protected/geo/identity fields added to `ALLOWED_FEATURES` or `FEATURE_NAMES`
- [ ] `cap_exploration` preserves full support when ethics cap applied
- [ ] `EthicalPolicy` used on user-facing demo/serving paths unless explicitly testing raw policy

**Source:** knowledge/ethics/rules.md, decisions/2026-06-01-feature-allow-list.md

**Last triggered:** never

---

## Category: API contract

**Severity:** blocking

**Criteria:**

- [ ] `/recommend` returns `decision_id`, `action`, `propensity`, `q_values`
- [ ] `/feedback` returns 204; unknown `decision_id` → 404
- [ ] `/route` returns stops, dropped, `total_time_s`, `total_profit`
- [ ] `build_app(orchestrator)` factory used in tests (no production disk side effects)

**Source:** tests/test_api.py, tests/test_e2e.py

**Last triggered:** never

---

## Category: Event store integrity

**Severity:** blocking

**Criteria:**

- [ ] No UPDATE/DELETE added to event store schema or accessors
- [ ] `ProspectContext` round-trips through JSON in store
- [ ] Latest outcome wins on reload when multiple outcome rows exist

**Source:** knowledge/nba-core/rules.md R3, tests/test_store.py

**Last triggered:** 2026-06-22 (drift demo must not unlink shared production EventStore)

**Severity:** warning

**Criteria:**

- [ ] Change is minimal — no unrelated refactors or drive-by fixes
- [ ] Matches existing naming, types, import style, and documentation level
- [ ] New public interfaces use existing `Protocol` patterns where applicable

**Source:** user coding principles, ARCHITECTURE.md §2 pluggable interfaces

**Last triggered:** never

---

## Category: Documentation

**Severity:** warning

**Criteria:**

- [ ] User-visible behavior changes reflected in README, ARCHITECTURE, or docs/ when non-trivial
- [ ] New decisions logged in `/decisions/` when replacing architectural choices
- [ ] Insights from task extracted to relevant `/knowledge/{domain}/` files

**Source:** AGENTS.md knowledge architecture

**Last triggered:** 2026-06-18 (Phase 9 + 17 — README/ARCHITECTURE/PLAN/docs updated, two decisions
logged, dataset-eval knowledge domain added)

---

## Always-check (promoted after 3+ triggers)

_None yet — promote criteria here after 3+ real catches._

---

## Prune candidates (never triggered after 10+ evaluations)

_None yet — first review after 10 task evaluations._
