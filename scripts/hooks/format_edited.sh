#!/usr/bin/env bash
#
# Cursor `afterFileEdit` hook: format the file the agent just edited.
# Payload arrives as JSON on stdin; we read .file_path and run ruff format on Python files.
# This is observational for the agent (afterFileEdit cannot block), so it just keeps the tree
# tidy and reduces what no_mistakes.sh has to fix later.
#
set -uo pipefail
payload="$(cat)"
if command -v jq >/dev/null 2>&1; then
  fp="$(printf '%s' "$payload" | jq -r '.file_path // .filePath // empty' 2>/dev/null)"
else
  fp="$(printf '%s' "$payload" | grep -oE '"file_?[Pp]ath"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*"([^"]+)"$/\1/')"
fi
[[ -n "${fp:-}" && -f "$fp" ]] || exit 0
case "$fp" in
  *.py)
    if command -v uv >/dev/null 2>&1; then uv run ruff format "$fp" >/dev/null 2>&1 || true
    elif command -v ruff >/dev/null 2>&1; then ruff format "$fp" >/dev/null 2>&1 || true
    fi
    ;;
esac
exit 0
