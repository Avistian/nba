#!/usr/bin/env python3
"""Backward-compatible entrypoint → flow/bin/agent-verify.py"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "flow/bin/agent-verify.py"), run_name="__main__")
