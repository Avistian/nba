#!/usr/bin/env bash
#
# overnight.sh — fan out gnhf + no-mistakes across one or more plan files (parallel worktrees).
#
# Typical flow after /to-issues:
#   1. Write plans/overnight/issue-101.md (+ optional .verify.json) per slice
#   2. Run this before bed
#   3. Review artifacts/overnight/<run>/summary.json in the morning
#
# Usage:
#   scripts/overnight.sh --job plans/overnight/issue-101.md [--job ...]
#   scripts/overnight.sh --parallel 2 --budget-usd 20 --max-steps 25 \
#     --job plans/overnight/issue-101.md --job plans/overnight/issue-102.md
#
# Options:
#   --job FILE              Plan file for gnhf (-f). Repeat for multiple jobs.
#   --verify-spec FILE      Passed to no_mistakes after gnhf (per job if same flag order:
#                           use --job plan:verify.json syntax below)
#   --job PLAN[:VERIFY]     Shorthand: plan path, optional verify spec after colon
#   --parallel N            Max concurrent gnhf runs (default: 1)
#   --budget-usd N          Per-job gnhf budget (default: 15)
#   --max-steps N           Per-job gnhf max steps (default: 25)
#   --validate CMD          Gate after each gnhf step (default: make check)
#   --base-context FILE     gnhf base context (default: AGENTS.md)
#   --base BRANCH           no_mistakes rebase target (default: main)
#   --skip-no-mistakes      Only run gnhf, skip validation pipeline
#   --dry-run               Print what would run
#   -h, --help
#
set -uo pipefail

die() { echo "overnight: $*" >&2; exit 1; }
log() { echo "[overnight $(date +%H:%M:%S)] $*"; }

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not in a git repo"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GNHF="$SCRIPT_DIR/gnhf.sh"
NM="$SCRIPT_DIR/no_mistakes.sh"

JOBS=()          # plan paths
VERIFY_SPECS=()  # parallel array, empty = auto-detect .cursor/capability-verify.json
PARALLEL=1
BUDGET=15
MAX_STEPS=25
VALIDATE="make check"
BASE_CONTEXT=""
BASE_BRANCH="main"
SKIP_NM=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      spec="${2:?}"
      if [[ "$spec" == *:* ]]; then
        JOBS+=("${spec%%:*}")
        VERIFY_SPECS+=("${spec#*:}")
      else
        JOBS+=("$spec")
        VERIFY_SPECS+=("")
      fi
      shift 2
      ;;
    --verify-spec) shift 2 ;; # legacy global — use --job plan:verify
    --parallel) PARALLEL="${2:?}"; shift 2 ;;
    --budget-usd) BUDGET="${2:?}"; shift 2 ;;
    --max-steps) MAX_STEPS="${2:?}"; shift 2 ;;
    --validate) VALIDATE="${2:?}"; shift 2 ;;
    --base-context) BASE_CONTEXT="${2:?}"; shift 2 ;;
    --base) BASE_BRANCH="${2:?}"; shift 2 ;;
    --skip-no-mistakes) SKIP_NM=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ ${#JOBS[@]} -gt 0 ]] || die "provide at least one --job PLAN.md"
[[ -x "$GNHF" ]] || die "missing $GNHF"
command -v git >/dev/null || die "git not found"
[[ -n "$(git -C "$ROOT" status --porcelain)" ]] && die "main worktree is dirty; commit or stash first"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$ROOT/artifacts/overnight/$RUN_ID"
mkdir -p "$RUN_DIR"
log "run dir: $RUN_DIR"

# --- helpers ------------------------------------------------------------------
slug_from_plan() {
  basename "$1" .md | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | cut -c1-32
}

run_gnhf_in_dir() {
  local plan="$1" workdir="$2" job_slug="$3"
  local -a args=(-f "$plan" --validate "$VALIDATE" --max-steps "$MAX_STEPS" --budget-usd "$BUDGET"
    --branch-prefix "overnight/${job_slug}-")
  [[ -n "$BASE_CONTEXT" ]] && args+=(--base-context "$BASE_CONTEXT")
  log "gnhf start: $job_slug in $workdir"
  (cd "$workdir" && "$GNHF" "${args[@]}") >"$RUN_DIR/${job_slug}-gnhf.log" 2>&1
}

run_no_mistakes_in_dir() {
  local workdir="$1" job_slug="$2" verify="$3"
  local -a args=(-y --base "$BASE_BRANCH" --no-push)
  [[ -n "$verify" && -f "$workdir/$verify" ]] && args+=(--verify-spec "$verify")
  log "no-mistakes start: $job_slug"
  (cd "$workdir" && "$NM" "${args[@]}") >"$RUN_DIR/${job_slug}-nm.log" 2>&1
}

# --- schedule jobs ------------------------------------------------------------
PIDS=()
JOB_SLUGS=()
i=0
for plan in "${JOBS[@]}"; do
  [[ -f "$ROOT/$plan" ]] || [[ -f "$plan" ]] || die "plan not found: $plan"
  plan_abs="$([[ "$plan" = /* ]] && echo "$plan" || echo "$ROOT/$plan")"
  slug="$(slug_from_plan "$plan_abs")"
  JOB_SLUGS+=("$slug")

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY: gnhf -f $plan_abs (slug=$slug, verify=${VERIFY_SPECS[$i]:-auto})"
    i=$((i+1))
    continue
  fi

  while [[ $(jobs -rp 2>/dev/null | wc -l) -ge $PARALLEL ]]; do
    sleep 5
  done

  if [[ ${#JOBS[@]} -eq 1 ]]; then
    workdir="$ROOT"
    log "single-job mode: main worktree"
    run_plan="$plan_abs"
  else
    workdir="$ROOT/../$(basename "$ROOT")-overnight-$slug"
    if [[ ! -d "$workdir" ]]; then
      git -C "$ROOT" worktree add -b "overnight/$slug" "$workdir" "$BASE_BRANCH" >/dev/null 2>&1 \
        || git -C "$ROOT" worktree add "$workdir" "$BASE_BRANCH" >/dev/null 2>&1 \
        || die "could not create worktree at $workdir"
    fi
    cp "$plan_abs" "$workdir/plan.md"
    run_plan="$workdir/plan.md"
  fi

  verify="${VERIFY_SPECS[$i]:-}"
  verify_arg=""
  if [[ -n "$verify" ]]; then
    vsrc="$([[ "$verify" = /* ]] && echo "$verify" || echo "$ROOT/$verify")"
    [[ -f "$vsrc" ]] || die "verify spec not found: $verify"
    if [[ "$workdir" != "$ROOT" ]]; then
      mkdir -p "$workdir/$(dirname "$verify")"
      cp "$vsrc" "$workdir/$verify"
    fi
    verify_arg="$verify"
  fi

  (
    rc=0
    run_gnhf_in_dir "$run_plan" "$workdir" "$slug" || rc=1
    if [[ $rc -eq 0 && "$SKIP_NM" != "1" ]]; then
      run_no_mistakes_in_dir "$workdir" "$slug" "$verify_arg" || rc=$?
    fi
    echo "$rc" >"$RUN_DIR/${slug}.exit"
  ) &
  PIDS+=($!)
  i=$((i+1))
done

# --- wait ---------------------------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
  log "dry-run complete (${#JOBS[@]} job(s))"
  exit 0
fi

log "waiting for ${#PIDS[@]} job(s) (parallel=$PARALLEL)..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

fail=0
{
  echo "{"
  echo "  \"run_id\": \"$RUN_ID\","
  echo "  \"jobs\": ["
} >"$RUN_DIR/summary.json"

j=0
for slug in "${JOB_SLUGS[@]}"; do
  ec=1
  [[ -f "$RUN_DIR/${slug}.exit" ]] && ec="$(cat "$RUN_DIR/${slug}.exit")"
  case "$ec" in
    0) st="pass"; log "$slug: PASS" ;;
    2) st="needs_human"; log "$slug: NEEDS HUMAN"; fail=2 ;;
    *) st="fail"; log "$slug: FAIL"; [[ "$fail" -ne 2 ]] && fail=1 ;;
  esac
  [[ $j -gt 0 ]] && echo "," >>"$RUN_DIR/summary.json"
  printf '    {"slug": "%s", "status": "%s", "exit": %s}' "$slug" "$st" "$ec" >>"$RUN_DIR/summary.json"
  j=$((j+1))
done
echo "" >>"$RUN_DIR/summary.json"
echo "  ]" >>"$RUN_DIR/summary.json"
echo "}" >>"$RUN_DIR/summary.json"

cat "$RUN_DIR/summary.json"
[[ "$fail" -eq 0 ]] && log "all jobs passed — $RUN_DIR" && exit 0
[[ "$fail" -eq 2 ]] && die "one or more jobs need human review — $RUN_DIR" 2
die "one or more jobs failed — $RUN_DIR"
