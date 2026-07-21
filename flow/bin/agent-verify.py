#!/usr/bin/env python3
"""
Agent-driven capability verification.

For each capability in a spec, runs two FRESH agent invocations:

  1. EXERCISE — the agent actually tries the capability (shell, APIs, browser MCP, custom
     scripts, whatever the repo needs). It must write artifacts under output_dir/<cap-id>/.
  2. JUDGE — a separate agent, with no exercise-session bias, reads only the acceptance
     criteria, exercise summary, and artifact evidence. Returns structured pass/fail/needs_human.

This is the versatile E2E layer: not limited to `npm run` or fixed smoke commands. Novel
capabilities are verified by an agent that exercises them and a second agent that judges evidence.

Usage:
  scripts/agent_verify.py --spec .cursor/capability-verify.json
  scripts/agent_verify.py --capability "User can paste images" --acceptance "Thumbnail appears"
  scripts/agent_verify.py --spec spec.json --base main --model gpt-5.4-high

Exit codes:
  0 — all required capabilities passed
  1 — one or more failed verification
  2 — needs_human escalation (ambiguous or could not exercise)

Environment:
  AGENT_BIN          Cursor CLI agent (default: agent)
  AGENT_VERIFY_MOCK  pass | fail | needs_human — bypass real agent (for tests)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGENT_BIN = os.environ.get("AGENT_BIN", "agent")
RESULT_KEYS = ("result", "text", "response")
DEFAULT_OUTPUT = "artifacts/capability-verify"


@dataclass
class StepResult:
    capability_id: str
    phase: str
    status: str
    score: float | None = None
    findings: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    raw: str = ""


def log(msg: str) -> None:
    print(f"[agent-verify] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    print(f"agent-verify: {msg}", file=sys.stderr)
    raise SystemExit(code)


def extract_agent_text(raw: str) -> str:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for key in RESULT_KEYS:
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            return json.dumps(data)
    except json.JSONDecodeError:
        pass
    return raw


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def run_agent(prompt: str, *, force: bool = False, model: str | None = None) -> str:
    mock = os.environ.get("AGENT_VERIFY_MOCK", "").strip().lower()
    if mock:
        return _mock_agent(prompt, mock)

    cmd = [AGENT_BIN, "-p", "--output-format", "json"]
    if force:
        cmd.append("--force")
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=3600)
    except FileNotFoundError:
        die(f"agent binary not found: {AGENT_BIN!r} (install Cursor CLI or set AGENT_BIN)")

    raw = proc.stdout or proc.stderr or ""
    if proc.returncode != 0 and not raw.strip():
        die(f"agent exited {proc.returncode} with no output")
    return extract_agent_text(raw)


def _mock_agent(prompt: str, mode: str) -> str:
    p = prompt.lower()
    if "independent judge" in p or "you are an independent judge" in p:
        verdict = {
            "pass": {"status": "pass", "score": 1.0, "findings": ["mock pass"], "ambiguous": []},
            "fail": {"status": "fail", "score": 0.0, "findings": ["mock fail"], "ambiguous": []},
            "needs_human": {
                "status": "needs_human",
                "score": 0.5,
                "findings": [],
                "ambiguous": ["mock: product decision needed"],
            },
        }[mode]
        return json.dumps(verdict)
    # exercise phase
    cap_dir = os.environ.get("AGENT_VERIFY_MOCK_DIR", "/tmp/agent-verify-mock")
    Path(cap_dir).mkdir(parents=True, exist_ok=True)
    summary = {
        "exercised": mode != "fail",
        "artifacts": [f"{cap_dir}/exercise.log"],
        "blockers": [] if mode != "needs_human" else ["cannot automate camera on CI"],
        "notes": f"mock exercise ({mode})",
    }
    Path(f"{cap_dir}/exercise.log").write_text("mock exercise output\n", encoding="utf-8")
    Path(f"{cap_dir}/exercise-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return json.dumps(summary)


def git_diff(base: str) -> str:
    for ref in (f"origin/{base}", base, "HEAD~5"):
        proc = subprocess.run(
            ["git", "diff", f"{ref}...HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout[:120_000]
    proc = subprocess.run(["git", "show", "--format=", "HEAD"], capture_output=True, text=True, check=False)
    return (proc.stdout or "")[:120_000]


def read_snippets(cap_dir: Path, limit: int = 12_000) -> str:
    parts: list[str] = []
    used = 0
    for path in sorted(cap_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".png", ".jpg", ".gif", ".webp", ".pdf", ".db"}:
            parts.append(f"[binary artifact] {path.relative_to(cap_dir)}")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunk = text[:4000]
        rel = path.relative_to(cap_dir)
        parts.append(f"--- {rel} ---\n{chunk}")
        used += len(chunk)
        if used >= limit:
            parts.append("... (truncated)")
            break
    return "\n\n".join(parts) if parts else "(no readable artifacts)"


def run_shell(cmd: str | None) -> None:
    if not cmd:
        return
    log(f"setup/teardown: {cmd[:80]}...")
    proc = subprocess.run(cmd, shell=True, check=False)
    if proc.returncode != 0:
        die(f"shell step failed ({proc.returncode}): {cmd}")


def exercise_capability(
    cap: dict[str, Any],
    cap_dir: Path,
    *,
    diff: str,
    model: str | None,
) -> StepResult:
    cap_id = cap["id"]
    cap_dir.mkdir(parents=True, exist_ok=True)
    desc = cap.get("description", cap_id)
    exercise = cap.get("exercise", {})
    hints = exercise.get("hints", cap.get("hints", []))
    if isinstance(hints, str):
        hints = [hints]
    tools = exercise.get("tools_allowed", ["shell", "read files", "browser MCP if configured"])
    extra = exercise.get("prompt_extra", "")

    prompt = f"""You are a QA engineer EXERCISING a new capability before it ships.
You must actually run commands / use tools — do not guess.

# CAPABILITY
id: {cap_id}
description: {desc}

# HOW TO EXERCISE (hints — adapt to this repo)
{chr(10).join(f"- {h}" for h in hints) if hints else "- Discover how to exercise from the diff and repo docs"}

# TOOLS YOU MAY USE
{json.dumps(tools)}

# CODE CHANGES (diff vs base)
```diff
{diff[:80_000]}
```

# OUTPUT DIRECTORY (write evidence here)
{cap_dir.resolve()}

# REQUIRED ARTIFACTS
1. Write step-by-step notes to: {cap_dir}/exercise.log
2. Save command output, API responses, screenshots, recordings — whatever proves you tried it.
3. When finished, write {cap_dir}/exercise-summary.json with STRICT JSON:
{{"exercised": true|false, "artifacts": ["paths relative to {cap_dir}"], "blockers": [], "notes": "..."}}

If you cannot exercise (missing hardware, auth, ambiguous product behavior), set exercised=false
and list blockers. Do NOT claim success without evidence files.

{extra}
"""
    if os.environ.get("AGENT_VERIFY_MOCK"):
        os.environ["AGENT_VERIFY_MOCK_DIR"] = str(cap_dir.resolve())
    raw = run_agent(prompt, force=True, model=model)
    summary_path = cap_dir / "exercise-summary.json"
    summary = extract_json_object(raw)
    if summary and not summary_path.exists():
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = summary or {}
    else:
        summary = summary or {"exercised": False, "blockers": ["no exercise-summary.json"], "artifacts": []}

    blockers = summary.get("blockers") or []
    exercised = bool(summary.get("exercised"))
    artifacts = [str(a) for a in summary.get("artifacts", [])]

    if blockers and not exercised:
        return StepResult(
            capability_id=cap_id,
            phase="exercise",
            status="needs_human",
            findings=[summary.get("notes", "exercise blocked")],
            ambiguous=[str(b) for b in blockers],
            artifacts=artifacts,
            raw=raw,
        )
    if not exercised:
        return StepResult(
            capability_id=cap_id,
            phase="exercise",
            status="fail",
            findings=["agent did not exercise capability"],
            artifacts=artifacts,
            raw=raw,
        )
    return StepResult(
        capability_id=cap_id,
        phase="exercise",
        status="exercised",
        artifacts=artifacts,
        raw=raw,
    )


def judge_capability(
    cap: dict[str, Any],
    cap_dir: Path,
    *,
    diff: str,
    model: str | None,
) -> StepResult:
    cap_id = cap["id"]
    desc = cap.get("description", cap_id)
    acceptance = cap.get("acceptance", [])
    if isinstance(acceptance, str):
        acceptance = [acceptance]
    min_score = float(cap.get("min_score", 1.0 if acceptance else 0.8))

    summary_path = cap_dir / "exercise-summary.json"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else "{}"
    snippets = read_snippets(cap_dir)

    prompt = f"""You are an INDEPENDENT JUDGE. You did NOT exercise this capability yourself.
You only see acceptance criteria, the exercise summary, and artifact contents below.

Return STRICT JSON only:
{{"status":"pass"|"fail"|"needs_human","score":0.0,"findings":["..."],"ambiguous":["..."]}}

Rules:
- pass: acceptance criteria are met WITH evidence in artifacts
- fail: clearly does not work or acceptance not met
- needs_human: product ambiguity, missing evidence, or exercise was incomplete
- score: 0.0-1.0 confidence the capability works as described

# CAPABILITY
id: {cap_id}
description: {desc}

# ACCEPTANCE CRITERIA
{chr(10).join(f"- {a}" for a in acceptance) if acceptance else "- Capability works as described in the diff"}

# EXERCISE SUMMARY
{summary_text}

# ARTIFACT SNIPPETS
{snippets}

# DIFF (for context only)
```diff
{diff[:40_000]}
```
"""
    raw = run_agent(prompt, force=False, model=model)
    verdict = extract_json_object(raw) or {}
    status = str(verdict.get("status", "fail")).lower()
    score = float(verdict.get("score", 0.0))
    findings = [str(f) for f in verdict.get("findings", [])]
    ambiguous = [str(a) for a in verdict.get("ambiguous", [])]

    if status == "pass" and score < min_score:
        status = "fail"
        findings.append(f"score {score} below min_score {min_score}")

    return StepResult(
        capability_id=cap_id,
        phase="judge",
        status=status,
        score=score,
        findings=findings,
        ambiguous=ambiguous,
        raw=raw,
    )


def load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def build_spec_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.spec:
        return load_spec(Path(args.spec))
    if not args.capability:
        die("provide --spec FILE or --capability DESCRIPTION")
    acceptance = args.acceptance or ["Capability works as described"]
    if isinstance(acceptance, list) and len(acceptance) == 1 and isinstance(acceptance[0], str) and "\n" in acceptance[0]:
        acceptance = [a.strip() for a in acceptance[0].split("\n") if a.strip()]
    hints = args.exercise_hint or []
    return {
        "version": 1,
        "name": "inline",
        "output_dir": args.output_dir,
        "capabilities": [
            {
                "id": args.capability_id or "capability",
                "description": args.capability,
                "acceptance": acceptance,
                "exercise": {"hints": hints},
            }
        ],
    }


def write_manifest(out_dir: Path, spec: dict[str, Any], results: list[StepResult]) -> Path:
    manifest = {
        "version": 1,
        "name": spec.get("name", "verify"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "capabilities": [
            {
                "id": r.capability_id,
                "phase": r.phase,
                "status": r.status,
                "score": r.score,
                "findings": r.findings,
                "ambiguous": r.ambiguous,
                "artifacts": r.artifacts,
            }
            for r in results
        ],
        "overall": _overall_status(results),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _overall_status(results: list[StepResult]) -> str:
    judge_results = [r for r in results if r.phase == "judge"]
    if any(r.status == "needs_human" for r in results):
        return "needs_human"
    if any(r.status == "fail" for r in judge_results):
        return "fail"
    if judge_results and all(r.status == "pass" for r in judge_results):
        return "pass"
    return "fail"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-driven capability verification")
    parser.add_argument("--spec", help="JSON spec file (.cursor/capability-verify.json)")
    parser.add_argument("--capability", help="Inline capability description")
    parser.add_argument("--capability-id", default="capability")
    parser.add_argument("--acceptance", action="append", help="Acceptance criterion (repeatable)")
    parser.add_argument("--exercise-hint", action="append", help="Hint for exercise agent")
    parser.add_argument("--base", default="main", help="Base branch for diff context")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-exercise", action="store_true", help="Judge existing artifacts only")
    args = parser.parse_args()

    spec = build_spec_from_args(args)
    out_dir = Path(spec.get("output_dir", args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    run_shell(spec.get("setup", {}).get("shell") if isinstance(spec.get("setup"), dict) else spec.get("setup"))

    diff = git_diff(args.base)
    results: list[StepResult] = []

    try:
        for cap in spec.get("capabilities", []):
            if "id" not in cap:
                die("each capability needs an id")
            cap_dir = out_dir / cap["id"]
            log(f"capability: {cap['id']}")

            if not args.skip_exercise:
                ex = exercise_capability(cap, cap_dir, diff=diff, model=args.model)
                results.append(ex)
                if ex.status == "needs_human":
                    manifest = write_manifest(out_dir, spec, results)
                    die(
                        f"{cap['id']}: exercise needs human — {ex.ambiguous}\nmanifest: {manifest}",
                        2,
                    )
                if ex.status == "fail":
                    manifest = write_manifest(out_dir, spec, results)
                    die(f"{cap['id']}: exercise failed\nmanifest: {manifest}", 1)

            judge = judge_capability(cap, cap_dir, diff=diff, model=args.model)
            results.append(judge)
            log(f"  judge: {judge.status} (score={judge.score})")

            if judge.status == "needs_human":
                manifest = write_manifest(out_dir, spec, results)
                die(
                    f"{cap['id']}: judge escalates — {judge.ambiguous}\nmanifest: {manifest}",
                    2,
                )
            if judge.status != "pass":
                manifest = write_manifest(out_dir, spec, results)
                die(f"{cap['id']}: judge FAIL — {judge.findings}\nmanifest: {manifest}", 1)

    finally:
        teardown = spec.get("teardown", {})
        shell = teardown.get("shell") if isinstance(teardown, dict) else teardown
        run_shell(shell)

    manifest = write_manifest(out_dir, spec, results)
    log(f"PASS — manifest: {manifest}")
    print(manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
