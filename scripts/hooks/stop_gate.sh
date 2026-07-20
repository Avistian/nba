#!/usr/bin/env bash
#
# Cursor `stop` hook: fast quality gate + E2E-evidence reminder when the agent finishes.
#
# Default mode is OBSERVE (never blocks) so it can't frustrate you while you tune it. Once you
# trust it, set NM_STOP_BLOCK=1 to make a failing lint/type gate return exit code 2 (blocks the
# stop on supported configurations). See knowledge/agentic-workflow/hypotheses.md (H1).
#
set -uo pipefail
BLOCK="${NM_STOP_BLOCK:-0}"

fail=0
if command -v uv >/dev/null 2>&1; then
  uv run ruff check . >/tmp/stop_ruff.log 2>&1 || fail=1
fi

if [[ "$fail" == "1" ]]; then
  echo "stop-gate: lint issues detected (see /tmp/stop_ruff.log). Fix before finishing." >&2
fi

# Nudge toward the trust guardrail: end-to-end evidence, not just green unit tests.
if [[ ! -d artifacts/no-mistakes ]] || ! ls artifacts/no-mistakes/evidence-* >/dev/null 2>&1; then
  echo "stop-gate: no E2E evidence found under artifacts/no-mistakes/. Consider running scripts/no_mistakes.sh." >&2
fi

if [[ "$BLOCK" == "1" && "$fail" == "1" ]]; then
  exit 2
fi
exit 0
