#!/usr/bin/env bash
#
# gnhf.sh — "good night, have fun"
#
# A long-running, fresh-context orchestrator for tasks too big for a single agent context.
# Each step runs in a NEW `agent -p` invocation seeded with a shared base context plus the
# learnings accumulated in notes.md. Passing steps are committed; failing steps are rolled
# back (git reset --hard) and their failure is fed forward so the next step avoids it. A token
# budget and max-step count bound the cost. You wake up to a branch of clean commits + notes.md.
#
# Inspired by the `gnhf` tool in "An Ex-Meta L8's Agentic Engineering Setup" (ByteByteGo, 2026),
# re-hosted on the Cursor CLI (`agent -p`). Model-agnostic by design.
#
# Usage:
#   scripts/gnhf.sh "<goal>" [options]
#   scripts/gnhf.sh -f plan.md [options]
#
# Options:
#   -f, --file FILE        Read the goal/plan from FILE instead of the positional arg.
#   --validate CMD         Validation gate run after each step (default: "make check").
#   --base-context FILE    File whose contents seed every step (default: AGENTS.md if present).
#   --max-steps N          Stop after N steps (default: 20).
#   --budget-usd N         Stop once estimated spend exceeds N (default: 10).
#   --branch-prefix P      Branch name prefix (default: "gnhf/").
#   --model SLUG           Model passed to `agent --model` (default: CLI default).
#   -h, --help             Show this help.
#
# Environment overrides (also used by the test harness):
#   AGENT_BIN   Command used as the agent (default: "agent"). Must accept:
#               $AGENT_BIN -p --force --output-format json [--model SLUG] "<prompt>"
#   COST_JQ     jq expression to extract per-step USD cost from the agent's JSON
#               (default: '.cost_usd // .usage.total_cost_usd // 0').
#   RESULT_JQ   jq expression to extract the agent's text result
#               (default: '.result // .text // .response // .').
#
set -uo pipefail

die() { echo "gnhf: $*" >&2; exit 1; }
log() { echo "[gnhf $(date +%H:%M:%S)] $*"; }

AGENT_BIN="${AGENT_BIN:-agent}"
COST_JQ="${COST_JQ:-.cost_usd // .usage.total_cost_usd // 0}"
RESULT_JQ="${RESULT_JQ:-.result // .text // .response // .}"

GOAL=""
GOAL_FILE=""
VALIDATE="make check"
BASE_CONTEXT=""
MAX_STEPS=20
BUDGET_USD=10
BRANCH_PREFIX="gnhf/"
MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--file) GOAL_FILE="${2:?}"; shift 2 ;;
    --validate) VALIDATE="${2:?}"; shift 2 ;;
    --base-context) BASE_CONTEXT="${2:?}"; shift 2 ;;
    --max-steps) MAX_STEPS="${2:?}"; shift 2 ;;
    --budget-usd) BUDGET_USD="${2:?}"; shift 2 ;;
    --branch-prefix) BRANCH_PREFIX="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) GOAL="$1"; shift ;;
  esac
done

command -v git >/dev/null || die "git not found"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not a git repo"
command -v "$AGENT_BIN" >/dev/null 2>&1 || die "agent binary '$AGENT_BIN' not found (set AGENT_BIN or install the Cursor CLI)"

HAVE_JQ=0
command -v jq >/dev/null 2>&1 && HAVE_JQ=1

if [[ -n "$GOAL_FILE" ]]; then
  [[ -f "$GOAL_FILE" ]] || die "goal file not found: $GOAL_FILE"
  GOAL="$(cat "$GOAL_FILE")"
fi
[[ -n "$GOAL" ]] || die "no goal given (pass a positional goal or -f FILE)"

if [[ -z "$BASE_CONTEXT" && -f AGENTS.md ]]; then BASE_CONTEXT="AGENTS.md"; fi
BASE_TEXT=""
[[ -n "$BASE_CONTEXT" && -f "$BASE_CONTEXT" ]] && BASE_TEXT="$(cat "$BASE_CONTEXT")"

# Refuse to run on a dirty tree so rollback (git reset --hard) can never eat real work.
if [[ -n "$(git status --porcelain)" ]]; then
  die "working tree is dirty; commit or stash first (gnhf uses 'git reset --hard' to roll back failed steps)"
fi

slug() { echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | cut -c1-40 | sed 's/-*$//'; }
BRANCH="${BRANCH_PREFIX}$(slug "$GOAL")-$(date +%m%d%H%M)"
git checkout -b "$BRANCH" >/dev/null 2>&1 || die "could not create branch $BRANCH"
log "branch: $BRANCH"

NOTES="notes.md"
{
  echo "# gnhf run — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "**Goal:** $GOAL"
  echo
  echo "**Validation gate:** \`$VALIDATE\`  |  **Budget:** \$$BUDGET_USD  |  **Max steps:** $MAX_STEPS"
  echo
  echo "## Step log"
  echo
} > "$NOTES"

extract() { # $1=jqexpr  reads json from stdin
  if [[ "$HAVE_JQ" == "1" ]]; then jq -r "$1" 2>/dev/null; else cat; fi
}

spend="0"
add_spend() { spend="$(awk -v a="$spend" -v b="${1:-0}" 'BEGIN{printf "%.4f", a+b}')"; }
over_budget() { awk -v s="$spend" -v b="$BUDGET_USD" 'BEGIN{exit !(s>b)}'; }

step=0
status="max-steps-reached"
while (( step < MAX_STEPS )); do
  step=$((step+1))
  log "step $step/$MAX_STEPS (spent \$$spend / \$$BUDGET_USD)"

  learnings="$(sed -n '/## Step log/,$p' "$NOTES")"
  prompt="You are one step of an autonomous multi-step run working toward a GOAL.

# GOAL
$GOAL

# BASE CONTEXT (repo conventions)
$BASE_TEXT

# LEARNINGS SO FAR (previous steps, including failures to avoid)
$learnings

# YOUR TASK
Make the SINGLE next smallest change that moves toward the GOAL and keeps the repo valid.
Do not attempt the whole goal at once. When (and only when) the GOAL is fully met and the
validation gate would pass, print the exact token GNHF_DONE on its own line and make no edits."

  raw="$("$AGENT_BIN" -p --force --output-format json ${MODEL:+--model "$MODEL"} "$prompt" 2>&1)"
  result="$(printf '%s' "$raw" | extract "$RESULT_JQ")"
  cost="$(printf '%s' "$raw" | extract "$COST_JQ")"
  [[ "$cost" =~ ^[0-9]+([.][0-9]+)?$ ]] || cost="0"
  add_spend "$cost"

  if printf '%s' "$result" | grep -q 'GNHF_DONE'; then
    log "agent signaled GNHF_DONE"
    echo "- step $step: **DONE** signaled by agent (\$$cost)" >> "$NOTES"
    status="done"
    break
  fi

  if [[ -z "$(git status --porcelain)" ]]; then
    log "no changes produced this step; treating as stuck"
    echo "- step $step: no changes produced (\$$cost) — stopping" >> "$NOTES"
    status="stuck-no-changes"
    break
  fi

  if bash -c "$VALIDATE" >/tmp/gnhf_validate.log 2>&1; then
    msg="$(printf '%s' "$result" | head -1 | cut -c1-72)"
    [[ -z "$msg" ]] && msg="gnhf step $step"
    git add -A
    git commit -q -m "gnhf: $msg" || true
    log "step $step PASSED gate → committed"
    echo "- step $step: PASS → committed \"$msg\" (\$$cost)" >> "$NOTES"
  else
    log "step $step FAILED gate → rolling back"
    tail_err="$(tail -n 15 /tmp/gnhf_validate.log | sed 's/^/    /')"
    # reset reverts tracked files; clean removes untracked files the failed step created,
    # but must preserve notes.md (untracked until the final commit) so learnings accumulate.
    git reset --hard -q HEAD
    git clean -fdq -e "$NOTES"
    {
      echo "- step $step: FAIL → rolled back (\$$cost). Gate output tail:"
      echo "$tail_err"
    } >> "$NOTES"
  fi

  if over_budget; then
    log "budget exceeded (\$$spend > \$$BUDGET_USD) → stopping"
    status="budget-exceeded"
    break
  fi
done

{
  echo
  echo "## Summary"
  echo "- Status: **$status**"
  echo "- Steps run: $step"
  echo "- Estimated spend: \$$spend"
  echo "- Branch: \`$BRANCH\`"
} >> "$NOTES"

git add "$NOTES" >/dev/null 2>&1 || true
git commit -q -m "gnhf: run notes ($status)" >/dev/null 2>&1 || true

log "finished: status=$status steps=$step spend=\$$spend branch=$BRANCH"
echo "See $NOTES for the full log. Next: review the branch, then run scripts/no_mistakes.sh."
