"""Control-flow tests for scripts/agent_verify.py (mock agent, no CURSOR_API_KEY)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "agent_verify.py"


def _run_mock(mode: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.io"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-qm", "init"], cwd=repo, check=True)
    env = {
        **os.environ,
        "AGENT_VERIFY_MOCK": mode,
        "AGENT_BIN": "false",
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--capability", "demo cap", "--acceptance", "works"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def test_mock_pass(tmp_path: Path) -> None:
    proc = _run_mock("pass", tmp_path)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    manifest = list((tmp_path / "repo" / "artifacts" / "capability-verify").rglob("manifest.json"))
    assert manifest
    data = json.loads(manifest[0].read_text())
    assert data["overall"] == "pass"


def test_mock_fail(tmp_path: Path) -> None:
    proc = _run_mock("fail", tmp_path)
    assert proc.returncode == 1


def test_mock_needs_human(tmp_path: Path) -> None:
    proc = _run_mock("needs_human", tmp_path)
    assert proc.returncode == 2
