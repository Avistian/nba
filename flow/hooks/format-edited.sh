#!/usr/bin/env bash
# Cursor afterFileEdit hook — format edited file using .cursor/flow.json patterns.
set -uo pipefail
payload="$(cat)"
fp=""
if command -v jq >/dev/null 2>&1; then
  fp="$(printf '%s' "$payload" | jq -r '.file_path // .filePath // empty' 2>/dev/null)"
else
  fp="$(printf '%s' "$payload" | grep -oE '"file_?[Pp]ath"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*"([^"]+)"$/\1/')"
fi
[[ -n "${fp:-}" && -f "$fp" ]] || exit 0

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CFG="$ROOT/.cursor/flow.json"

run_cmd() {
  local cmd="$1"
  cmd="${cmd//\{file\}/$fp}"
  bash -c "$cmd" >/dev/null 2>&1 || true
}

if [[ -f "$CFG" ]] && command -v jq >/dev/null 2>&1; then
  while IFS= read -r pattern; do
    [[ "$fp" == $pattern ]] || continue
    cmd="$(jq -r --arg p "$pattern" '.format_on_edit[$p] // empty' "$CFG")"
    [[ -n "$cmd" && "$cmd" != "null" ]] && run_cmd "$cmd" && exit 0
  done < <(jq -r '.format_on_edit // {} | keys[]' "$CFG" 2>/dev/null)
fi

# Built-in fallbacks
case "$fp" in
  *.py)
    command -v uv >/dev/null && run_cmd "uv run ruff format {file}"
    command -v ruff >/dev/null && run_cmd "ruff format {file}"
    ;;
  *.ts|*.tsx|*.js|*.jsx|*.json|*.css|*.md)
    command -v npx >/dev/null && run_cmd "npx prettier --write {file}"
    ;;
esac
exit 0
