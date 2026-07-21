#!/usr/bin/env bash
# Bootstrap flow/ into a repo or install CLI symlinks globally.
set -euo pipefail

FLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
MODE="${1:-repo}"

install_repo() {
  echo "flow: bootstrapping repo at $REPO_ROOT"
  mkdir -p "$REPO_ROOT/.cursor"

  copy_if_missing() {
    local src="$1" dst="$2"
    if [[ -f "$dst" ]]; then echo "  keep $dst"
    else cp "$src" "$dst" && echo "  created $dst"; fi
  }

  copy_if_missing "$FLOW_ROOT/templates/flow.json.example" "$REPO_ROOT/.cursor/flow.json"
  copy_if_missing "$FLOW_ROOT/templates/capability-verify.example.json" "$REPO_ROOT/.cursor/capability-verify.json"
  # Prefer repo-specific example if present
  if [[ -f "$REPO_ROOT/.cursor/capability-verify.example.json" ]]; then
    if [[ ! -f "$REPO_ROOT/.cursor/capability-verify.json" ]] || \
       cmp -s "$REPO_ROOT/.cursor/capability-verify.json" "$FLOW_ROOT/templates/capability-verify.example.json" 2>/dev/null; then
      cp "$REPO_ROOT/.cursor/capability-verify.example.json" "$REPO_ROOT/.cursor/capability-verify.json"
      echo "  used .cursor/capability-verify.example.json"
    fi
  fi
  copy_if_missing "$FLOW_ROOT/templates/hooks.example.json" "$REPO_ROOT/.cursor/hooks.json"
  mkdir -p "$REPO_ROOT/.cursor/commands"
  copy_if_missing "$FLOW_ROOT/templates/no-mistakes.toml" "$REPO_ROOT/.cursor/commands/no-mistakes.toml"
  mkdir -p "$REPO_ROOT/plans/overnight"

  chmod +x "$FLOW_ROOT/bin/"* "$FLOW_ROOT/hooks/"*.sh 2>/dev/null || true

  echo ""
  echo "Done. Add to your shell profile (optional):"
  echo "  export PATH=\"$FLOW_ROOT/bin:\$PATH\""
  echo ""
  echo "Or run via: $FLOW_ROOT/bin/gnhf | no-mistakes | overnight"
  echo "Configure: edit $REPO_ROOT/.cursor/flow.json"
}

install_global() {
  local bindir="${INSTALL_DIR:-$HOME/.local/bin}"
  mkdir -p "$bindir"
  for tool in gnhf no-mistakes overnight; do
    ln -sf "$FLOW_ROOT/bin/$tool" "$bindir/flow-$tool"
    echo "  linked $bindir/flow-$tool"
  done
  ln -sf "$FLOW_ROOT/bin/agent-verify.py" "$bindir/flow-agent-verify"
  echo "Set FLOW_ROOT=$FLOW_ROOT in profile if flow/ moves."
}

case "$MODE" in
  repo) install_repo ;;
  global) install_global ;;
  *)
    echo "Usage: flow/install.sh [repo|global]"
    echo "  repo   — copy templates into .cursor/ (default)"
    echo "  global — symlink flow-* tools to ~/.local/bin"
    exit 1
    ;;
esac
