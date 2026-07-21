#!/usr/bin/env bash
#
# no_mistakes.sh — autonomous validation pipeline you run AFTER an agent finishes a change.
#
# Recreates the `no-mistakes` tool from "An Ex-Meta L8's Agentic Engineering Setup" on Cursor
# primitives (`agent -p` for fresh-context review/fixes). It:
#   1. shows the diffstat and (without -y) waits for your glance-gate confirmation
#   2. commits with a Conventional Commit message onto a descriptively named branch
#   3. rebases onto the latest base branch (agent resolves conflicts if any), re-validates
#   4. runs a FRESH-CONTEXT reviewer that returns {obvious_bugs, ambiguous}
#        - obvious bugs  -> auto-fixed as separate commits
#        - ambiguous / product-changing -> STOP and escalate to the human (exit 2)
#   5. runs the E2E command and requires real evidence (artifact) before continuing
#   6. formats/lints and closes doc gaps
#   7. pushes and prints the PR body (with an auto-fix log + Testing section)
#
# Trust guardrails (do not remove): fresh-context review, human escalation on ambiguity,
# forced E2E evidence, one-commit-per-fix auditability.
#
# Usage:  scripts/no_mistakes.sh [options]
#   -y, --yes            Skip the interactive glance gate.
#   --base BRANCH        Base branch to rebase onto (default: main).
#   --validate CMD       Fast gate (default: "make check").
#   --e2e CMD            Shell E2E command (legacy; use --verify-spec for agent verification).
#   --verify-spec FILE   Agent exercises + judges capabilities (see scripts/agent_verify.py).
#   --verify-capability DESC   Quick single-capability agent verify (repeatable).
#   --acceptance CRIT    Acceptance criterion for --verify-capability (repeatable).
#   --exercise-hint HINT Hint for the exercise agent (repeatable).
#   --no-push            Do everything except push / PR.
#   --no-rebase          Skip the rebase step.
#   -h, --help           Show help.
#
# Environment overrides (also used by the test harness):
#   AGENT_BIN, RESULT_JQ   (see gnhf.sh)
#   REVIEW_JQ_BUGS         jq for the obvious-bug list  (default: '.obvious_bugs // []')
#   REVIEW_JQ_AMBIG        jq for the ambiguous list    (default: '.ambiguous // []')
#
set -uo pipefail

die() { echo "no-mistakes: $*" >&2; exit 1; }
log() { echo "[no-mistakes $(date +%H:%M:%S)] $*"; }
esc() { echo; echo "==== ESCALATION (human decision required) ===="; echo "$*"; echo "=============================================="; exit 2; }

AGENT_BIN="${AGENT_BIN:-agent}"
RESULT_JQ="${RESULT_JQ:-.result // .text // .response // .}"
REVIEW_JQ_BUGS="${REVIEW_JQ_BUGS:-.obvious_bugs // []}"
REVIEW_JQ_AMBIG="${REVIEW_JQ_AMBIG:-.ambiguous // []}"

YES=0
BASE="main"
VALIDATE="make check"
E2E="make demo"
VERIFY_SPEC=""
VERIFY_CAP=""
VERIFY_ACCEPT=()
VERIFY_HINTS=()
PUSH=1
REBASE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) YES=1; shift ;;
    --base) BASE="${2:?}"; shift 2 ;;
    --validate) VALIDATE="${2:?}"; shift 2 ;;
    --e2e) E2E="${2:?}"; shift 2 ;;
    --verify-spec) VERIFY_SPEC="${2:?}"; shift 2 ;;
    --verify-capability) VERIFY_CAP="${2:?}"; shift 2 ;;
    --acceptance) VERIFY_ACCEPT+=("${2:?}"); shift 2 ;;
    --exercise-hint) VERIFY_HINTS+=("${2:?}"); shift 2 ;;
    --no-push) PUSH=0; shift ;;
    --no-rebase) REBASE=0; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

command -v git >/dev/null || die "git not found"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not a git repo"
command -v "$AGENT_BIN" >/dev/null 2>&1 || die "agent binary '$AGENT_BIN' not found"
HAVE_JQ=0; command -v jq >/dev/null 2>&1 && HAVE_JQ=1
extract() { if [[ "$HAVE_JQ" == "1" ]]; then jq -r "$1" 2>/dev/null; else cat; fi; }

# --- 1. Glance gate -----------------------------------------------------------
DIFFSTAT="$(git diff --stat HEAD; git status --porcelain)"
echo "----- change under review -----"
echo "$DIFFSTAT"
echo "-------------------------------"
if [[ "$YES" != "1" ]]; then
  read -r -p "Does this look roughly like what you asked for? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || die "aborted at glance gate"
fi

# --- 2. Commit onto a descriptive branch --------------------------------------
if [[ -n "$(git status --porcelain)" ]]; then
  cur="$(git rev-parse --abbrev-ref HEAD)"
  if [[ "$cur" == "$BASE" ]]; then
    slug="$("$AGENT_BIN" -p --output-format json "In 3-6 kebab-case words, name a git branch for this diff. Print ONLY the slug.
$DIFFSTAT" 2>/dev/null | extract "$RESULT_JQ" | tr -cs 'a-z0-9-' '-' | sed 's/^-*//;s/-*$//' | cut -c1-40)"
    [[ -z "$slug" ]] && slug="change-$(date +%m%d%H%M)"
    git checkout -b "feat/$slug" >/dev/null 2>&1 || die "could not create branch feat/$slug"
    log "created branch feat/$slug"
  fi
  msg="$("$AGENT_BIN" -p --output-format json "Write ONE Conventional Commit subject line (<=72 chars) for this diff. Print ONLY the line.
$DIFFSTAT" 2>/dev/null | extract "$RESULT_JQ" | head -1)"
  [[ -z "$msg" ]] && msg="chore: apply agent changes"
  git add -A && git commit -q -m "$msg"
  log "committed: $msg"
else
  log "no uncommitted changes; reviewing latest commit(s) on this branch"
fi

# --- 3. Rebase onto latest base ----------------------------------------------
if [[ "$REBASE" == "1" ]]; then
  log "fetching + rebasing onto $BASE"
  git fetch origin "$BASE" >/dev/null 2>&1 || log "warn: fetch failed (offline?) — rebasing on local $BASE"
  if ! git rebase "origin/$BASE" >/dev/null 2>&1 && ! git rebase "$BASE" >/dev/null 2>&1; then
    log "rebase conflict — asking agent to resolve"
    "$AGENT_BIN" -p --force --output-format json "Resolve the current git rebase conflicts consistent with both sides' intent, then stage the files. Do not run git rebase --continue." >/dev/null 2>&1 || true
    git add -A && git rebase --continue >/dev/null 2>&1 || esc "rebase conflicts could not be resolved automatically; resolve by hand."
  fi
  bash -c "$VALIDATE" >/tmp/nm_validate.log 2>&1 || esc "validation failed after rebase; see /tmp/nm_validate.log"
fi

# --- 4. Fresh-context peer review --------------------------------------------
log "fresh-context review"
DIFF="$(git diff "$BASE"...HEAD 2>/dev/null || git show --format= HEAD)"
review="$("$AGENT_BIN" -p --output-format json "You are a peer reviewer seeing ONLY this diff, with no memory of who wrote it.
Return STRICT JSON: {\"obvious_bugs\":[{\"file\":\"\",\"why\":\"\",\"fix\":\"\"}],\"ambiguous\":[{\"question\":\"\"}]}.
'obvious_bugs' = clearly-wrong, safe-to-auto-fix defects. 'ambiguous' = anything that changes
product behavior or needs a human judgement call. Diff:
$DIFF" 2>/dev/null)"

ambig="$(printf '%s' "$review" | extract "$REVIEW_JQ_AMBIG")"
bugs="$(printf '%s' "$review" | extract "$REVIEW_JQ_BUGS")"
ambig_n=0; bugs_n=0
if [[ "$HAVE_JQ" == "1" ]]; then
  ambig_n="$(printf '%s' "$review" | jq "($REVIEW_JQ_AMBIG) | length" 2>/dev/null || echo 0)"
  bugs_n="$(printf '%s' "$review" | jq "($REVIEW_JQ_BUGS) | length" 2>/dev/null || echo 0)"
fi

FIXLOG=""
if [[ "${bugs_n:-0}" -gt 0 ]]; then
  log "auto-fixing $bugs_n obvious bug(s)"
  "$AGENT_BIN" -p --force --output-format json "Fix ONLY these clearly-wrong bugs; change nothing else:
$bugs" >/dev/null 2>&1 || true
  if [[ -n "$(git status --porcelain)" ]]; then
    bash -c "$VALIDATE" >/tmp/nm_validate.log 2>&1 || esc "auto-fix broke the gate; see /tmp/nm_validate.log"
    git add -A && git commit -q -m "fix: address reviewer-flagged bugs"
    FIXLOG="$bugs"
  fi
fi

if [[ "${ambig_n:-0}" -gt 0 ]]; then
  esc "Reviewer flagged $ambig_n ambiguous / product-changing item(s). NOT auto-fixed:
$ambig"
fi

# --- 5. Capability verification (agent exercises + judges) or shell E2E ------------
mkdir -p artifacts/no-mistakes
EV=""
MANIFEST=""
VERIFY_MODE="shell"

if [[ -n "$VERIFY_SPEC" || -n "$VERIFY_CAP" ]]; then
  VERIFY_MODE="agent"
  log "agent capability verification"
  AV=(python3 "$(git rev-parse --show-toplevel)/scripts/agent_verify.py" --base "$BASE")
  [[ -n "$VERIFY_SPEC" ]] && AV+=(--spec "$VERIFY_SPEC")
  if [[ -n "$VERIFY_CAP" ]]; then
    AV+=(--capability "$VERIFY_CAP")
    for a in "${VERIFY_ACCEPT[@]}"; do AV+=(--acceptance "$a"); done
    for h in "${VERIFY_HINTS[@]}"; do AV+=(--exercise-hint "$h"); done
  fi
  if ! "${AV[@]}" >artifacts/no-mistakes/agent-verify.log 2>&1; then
    rc=$?
    esc "Agent capability verification failed (exit $rc). Log:
$(tail -n 30 artifacts/no-mistakes/agent-verify.log)"
  fi
  MANIFEST="$(find artifacts/capability-verify -name manifest.json 2>/dev/null | head -1)"
  [[ -n "$MANIFEST" && -f "$MANIFEST" ]] || esc "Agent verify passed but no manifest.json found"
  EV="artifacts/no-mistakes/agent-verify.log"
  log "capability verification passed — manifest: $MANIFEST"
elif [[ -f .cursor/capability-verify.json ]]; then
  VERIFY_MODE="agent"
  log "found .cursor/capability-verify.json — running agent verification"
  if ! python3 "$(git rev-parse --show-toplevel)/scripts/agent_verify.py" \
      --spec .cursor/capability-verify.json --base "$BASE" \
      >artifacts/no-mistakes/agent-verify.log 2>&1; then
    rc=$?
    esc "Agent capability verification failed (exit $rc). Log:
$(tail -n 30 artifacts/no-mistakes/agent-verify.log)"
  fi
  MANIFEST="$(find artifacts/capability-verify -name manifest.json 2>/dev/null | head -1)"
  EV="artifacts/no-mistakes/agent-verify.log"
  log "capability verification passed — manifest: $MANIFEST"
else
  log "collecting shell E2E evidence: $E2E"
  EV="artifacts/no-mistakes/evidence-$(date +%Y%m%d-%H%M%S).log"
  if ! bash -c "$E2E" >"$EV" 2>&1; then
    esc "E2E command failed; evidence at $EV (tail below):
$(tail -n 20 "$EV")"
  fi
  [[ -s "$EV" ]] || esc "E2E produced no output — refusing to claim it works. See $E2E."
  log "E2E evidence saved: $EV"
fi

# --- 6. Format + doc gaps -----------------------------------------------------
bash -c "make fmt" >/dev/null 2>&1 || true
if [[ -n "$(git status --porcelain)" ]]; then git add -A && git commit -q -m "style: apply formatter"; fi

# --- 7. Push + PR body --------------------------------------------------------
PRBODY="artifacts/no-mistakes/pr-body.md"
{
  echo "## Summary"
  echo "Automated pipeline (no_mistakes.sh) processed this change."
  echo
  echo "## Auto-fixes applied"
  if [[ -n "$FIXLOG" ]]; then echo '```'; echo "$FIXLOG"; echo '```'; else echo "_none_"; fi
  echo
  echo "## Testing (capability verification)"
  if [[ "$VERIFY_MODE" == "agent" ]]; then
    echo "Agent exercised and judged capabilities. Manifest: \`${MANIFEST:-artifacts/capability-verify/manifest.json}\`"
    echo
    if [[ -n "$MANIFEST" && -f "$MANIFEST" ]]; then
      echo '```json'
      cat "$MANIFEST"
      echo '```'
    fi
    echo
    echo "Exercise log tail (\`$EV\`):"
  else
    echo "Ran shell E2E: \`$E2E\`; output at \`$EV\`."
  fi
  echo '```'
  tail -n 20 "$EV"
  echo '```'
} > "$PRBODY"

if [[ "$PUSH" == "1" ]]; then
  br="$(git rev-parse --abbrev-ref HEAD)"
  git push -u origin "$br" >/dev/null 2>&1 || log "warn: push failed (auth/offline?) — push manually"
  if command -v gh >/dev/null 2>&1; then
    log "opening PR via gh"
    gh pr create --fill --body-file "$PRBODY" 2>/dev/null || log "gh pr create failed — open the PR manually with $PRBODY"
  else
    log "gh not installed — PR body written to $PRBODY (open the PR manually)"
  fi
else
  log "--no-push set; PR body at $PRBODY"
fi

log "done. Review the PR; merging stays a human step."
