#!/usr/bin/env bash
# flow/lib/config.sh — load per-repo config + defaults for all flow tools.
#
# Usage (from any flow/bin script):
#   source "$(dirname "$0")/../lib/config.sh"
#   flow_init
#   echo "$FLOW_VALIDATE"

flow_init() {
  local _lib
  _lib="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  FLOW_ROOT="$(cd "$_lib/.." && pwd)"
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  FLOW_CONFIG="${FLOW_CONFIG:-$REPO_ROOT/.cursor/flow.json}"

  # shellcheck source=detect.sh
  source "$FLOW_ROOT/lib/detect.sh"
  _flow_detect

  FLOW_BASE_BRANCH="main"
  FLOW_VALIDATE="$FLOW_DETECT_VALIDATE"
  FLOW_E2E="$FLOW_DETECT_E2E"
  FLOW_FORMAT="$FLOW_DETECT_FORMAT"
  FLOW_BASE_CONTEXT=""
  FLOW_VERIFY_SPEC=".cursor/capability-verify.json"
  FLOW_ARTIFACTS="artifacts"
  FLOW_GNHF_MAX_STEPS=20
  FLOW_GNHF_BUDGET=10
  FLOW_OVERNIGHT_PARALLEL=1
  FLOW_OVERNIGHT_BUDGET=15
  FLOW_OVERNIGHT_MAX_STEPS=25
  FLOW_AGENT_VERIFY="$FLOW_ROOT/bin/agent-verify.py"

  if [[ -f "$FLOW_CONFIG" ]] && command -v jq >/dev/null 2>&1; then
    FLOW_BASE_BRANCH="$(jq -r '.base_branch // empty' "$FLOW_CONFIG")"
    FLOW_VALIDATE="$(jq -r '.validate // empty' "$FLOW_CONFIG")"
    FLOW_E2E="$(jq -r '.e2e // empty' "$FLOW_CONFIG")"
    FLOW_FORMAT="$(jq -r '.format // empty' "$FLOW_CONFIG")"
    FLOW_BASE_CONTEXT="$(jq -r '.base_context // empty' "$FLOW_CONFIG")"
    FLOW_VERIFY_SPEC="$(jq -r '.capability_verify_spec // empty' "$FLOW_CONFIG")"
    FLOW_ARTIFACTS="$(jq -r '.artifacts_dir // empty' "$FLOW_CONFIG")"
    FLOW_GNHF_MAX_STEPS="$(jq -r '.gnhf.max_steps // empty' "$FLOW_CONFIG")"
    FLOW_GNHF_BUDGET="$(jq -r '.gnhf.budget_usd // empty' "$FLOW_CONFIG")"
    FLOW_OVERNIGHT_PARALLEL="$(jq -r '.overnight.parallel // empty' "$FLOW_CONFIG")"
    FLOW_OVERNIGHT_BUDGET="$(jq -r '.overnight.budget_usd // empty' "$FLOW_CONFIG")"
    FLOW_OVERNIGHT_MAX_STEPS="$(jq -r '.overnight.max_steps // empty' "$FLOW_CONFIG")"
    # jq returns "null" / empty — fall back
    [[ "$FLOW_BASE_BRANCH" == "null" || -z "$FLOW_BASE_BRANCH" ]] && FLOW_BASE_BRANCH="main"
    [[ "$FLOW_VALIDATE" == "null" || -z "$FLOW_VALIDATE" ]] && FLOW_VALIDATE="$FLOW_DETECT_VALIDATE"
    [[ "$FLOW_E2E" == "null" || -z "$FLOW_E2E" ]] && FLOW_E2E="$FLOW_DETECT_E2E"
    [[ "$FLOW_FORMAT" == "null" || -z "$FLOW_FORMAT" ]] && FLOW_FORMAT="$FLOW_DETECT_FORMAT"
    [[ "$FLOW_VERIFY_SPEC" == "null" || -z "$FLOW_VERIFY_SPEC" ]] && FLOW_VERIFY_SPEC=".cursor/capability-verify.json"
    [[ "$FLOW_ARTIFACTS" == "null" || -z "$FLOW_ARTIFACTS" ]] && FLOW_ARTIFACTS="artifacts"
    [[ "$FLOW_GNHF_MAX_STEPS" == "null" || -z "$FLOW_GNHF_MAX_STEPS" ]] && FLOW_GNHF_MAX_STEPS=20
    [[ "$FLOW_GNHF_BUDGET" == "null" || -z "$FLOW_GNHF_BUDGET" ]] && FLOW_GNHF_BUDGET=10
    [[ "$FLOW_OVERNIGHT_PARALLEL" == "null" || -z "$FLOW_OVERNIGHT_PARALLEL" ]] && FLOW_OVERNIGHT_PARALLEL=1
    [[ "$FLOW_OVERNIGHT_BUDGET" == "null" || -z "$FLOW_OVERNIGHT_BUDGET" ]] && FLOW_OVERNIGHT_BUDGET=15
    [[ "$FLOW_OVERNIGHT_MAX_STEPS" == "null" || -z "$FLOW_OVERNIGHT_MAX_STEPS" ]] && FLOW_OVERNIGHT_MAX_STEPS=25
  fi

  if [[ -z "$FLOW_BASE_CONTEXT" ]]; then
    for f in AGENTS.md CONTEXT.md README.md; do
      [[ -f "$REPO_ROOT/$f" ]] && FLOW_BASE_CONTEXT="$f" && break
    done
  fi

  export FLOW_ROOT REPO_ROOT FLOW_CONFIG
  export FLOW_VALIDATE FLOW_E2E FLOW_FORMAT FLOW_BASE_BRANCH FLOW_BASE_CONTEXT
  export FLOW_VERIFY_SPEC FLOW_ARTIFACTS FLOW_AGENT_VERIFY
}
