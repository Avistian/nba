#!/usr/bin/env bash
# flow/lib/detect.sh — auto-detect validate / e2e / format commands for any repo.
# Sourced by config.sh. Sets FLOW_DETECT_VALIDATE, FLOW_DETECT_E2E, FLOW_DETECT_FORMAT.

_flow_detect() {
  FLOW_DETECT_VALIDATE=""
  FLOW_DETECT_E2E=""
  FLOW_DETECT_FORMAT=""

  if [[ -f Makefile ]]; then
    grep -qE '^check:' Makefile 2>/dev/null && FLOW_DETECT_VALIDATE="make check"
    grep -qE '^demo:' Makefile 2>/dev/null && FLOW_DETECT_E2E="make demo"
    grep -qE '^e2e:' Makefile 2>/dev/null && FLOW_DETECT_E2E="make e2e"
    grep -qE '^smoke:' Makefile 2>/dev/null && [[ -z "$FLOW_DETECT_E2E" ]] && FLOW_DETECT_E2E="make smoke"
    grep -qE '^fmt:' Makefile 2>/dev/null && FLOW_DETECT_FORMAT="make fmt"
    grep -qE '^format:' Makefile 2>/dev/null && [[ -z "$FLOW_DETECT_FORMAT" ]] && FLOW_DETECT_FORMAT="make format"
  fi

  if [[ -z "$FLOW_DETECT_VALIDATE" && -f package.json ]]; then
    command -v npm >/dev/null 2>&1 && FLOW_DETECT_VALIDATE="npm test"
    grep -q '"lint"' package.json 2>/dev/null && FLOW_DETECT_VALIDATE="npm run lint && npm test"
  fi
  if [[ -z "$FLOW_DETECT_E2E" && -f package.json ]]; then
    grep -q '"test:e2e"' package.json 2>/dev/null && FLOW_DETECT_E2E="npm run test:e2e"
    grep -q '"e2e"' package.json 2>/dev/null && [[ -z "$FLOW_DETECT_E2E" ]] && FLOW_DETECT_E2E="npm run e2e"
  fi
  if [[ -z "$FLOW_DETECT_FORMAT" && -f package.json ]]; then
    grep -q '"format"' package.json 2>/dev/null && FLOW_DETECT_FORMAT="npm run format"
    grep -q '"fmt"' package.json 2>/dev/null && [[ -z "$FLOW_DETECT_FORMAT" ]] && FLOW_DETECT_FORMAT="npm run fmt"
  fi

  if [[ -z "$FLOW_DETECT_VALIDATE" && -f pyproject.toml ]]; then
    if command -v uv >/dev/null 2>&1; then
      FLOW_DETECT_VALIDATE="uv run pytest"
      FLOW_DETECT_FORMAT="uv run ruff format ."
    elif command -v pytest >/dev/null 2>&1; then
      FLOW_DETECT_VALIDATE="pytest"
    fi
  fi

  if [[ -z "$FLOW_DETECT_VALIDATE" && -f Cargo.toml ]]; then
    FLOW_DETECT_VALIDATE="cargo test && cargo clippy -- -D warnings"
  fi

  if [[ -z "$FLOW_DETECT_VALIDATE" && -f go.mod ]]; then
    FLOW_DETECT_VALIDATE="go test ./..."
  fi

  [[ -n "$FLOW_DETECT_VALIDATE" ]] || FLOW_DETECT_VALIDATE="echo 'Set validate in .cursor/flow.json' && exit 1"
  [[ -n "$FLOW_DETECT_E2E" ]] || FLOW_DETECT_E2E="echo 'Set e2e in .cursor/flow.json' && exit 1"
  [[ -z "$FLOW_DETECT_FORMAT" ]] && FLOW_DETECT_FORMAT="true"
}
