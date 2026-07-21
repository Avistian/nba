#!/usr/bin/env bash
# Cursor stop hook — lint reminder + E2E evidence nudge. Set FLOW_STOP_BLOCK=1 to block.
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CFG="$ROOT/.cursor/flow.json"
BLOCK="${FLOW_STOP_BLOCK:-0}"
fail=0

validate=""
[[ -f "$CFG" ]] && command -v jq >/dev/null && validate="$(jq -r '.validate // empty' "$CFG")"
[[ -z "$validate" || "$validate" == "null" ]] && validate="true"

bash -c "$validate" >/tmp/flow_stop.log 2>&1 || fail=1

if [[ ! -d "$ROOT/artifacts/capability-verify" ]] && [[ ! -d "$ROOT/artifacts/no-mistakes" ]]; then
  echo "flow stop-gate: no verification artifacts yet — run flow/bin/no-mistakes" >&2
fi

[[ $fail -eq 1 ]] && echo "flow stop-gate: validate failed — see /tmp/flow_stop.log" >&2
[[ "$BLOCK" == "1" && $fail -eq 1 ]] && exit 2
exit 0
