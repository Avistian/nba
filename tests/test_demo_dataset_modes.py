"""Guards for the dataset-aware demo: flat output stays deterministic after the oracle indirection,
and the relational mode runs end-to-end through the same machinery."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from nba.config import Settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_demo import run_demo  # noqa: E402


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        db_path=tmp_path / "events.db",
        n_bootstrap=2,
        **overrides,
    )


def _run(settings: Settings):
    return run_demo(
        settings=settings,
        n_logs=1500,
        shift=18,
        replan_every=6,
        seed=7,
        ope_max_rows=300,
        write=False,
    )


def test_flat_demo_is_deterministic(tmp_path: Path) -> None:
    """The flat path (default) must be byte-identical run-to-run after the oracle refactor."""
    a = _run(_settings(tmp_path / "a"))
    b = _run(_settings(tmp_path / "b"))
    assert a.settings is not None and a.settings.dataset_mode == "flat"  # default unchanged
    assert a.to_json() == b.to_json()


def test_relational_demo_runs_end_to_end(tmp_path: Path) -> None:
    report = _run(_settings(tmp_path, dataset_mode="relational"))
    assert report.n_decisions > 0
    assert report.min_propensity > 0.0  # overlap holds => OPE-valid
    assert report.expected_reward["oracle_best"] >= report.expected_reward["bandit"]


def test_relational_demo_is_deterministic(tmp_path: Path) -> None:
    a = _run(_settings(tmp_path / "a", dataset_mode="relational"))
    b = _run(_settings(tmp_path / "b", dataset_mode="relational"))
    assert a.to_json() == b.to_json()
