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

**Last triggered:** never (enforced continuously; no violation since Phase 8)

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

**Last triggered:** never

---

## Category: Scope & diff discipline

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

**Last triggered:** 2026-06-13 (AGENTS.md bootstrap)

---

## Always-check (promoted after 3+ triggers)

_None yet — promote criteria here after 3+ real catches._

---

## Prune candidates (never triggered after 10+ evaluations)

_None yet — first review after 10 task evaluations._
