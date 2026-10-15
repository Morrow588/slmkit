"""The training-run listing command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slmkit.cli import app


def _write_status(home: Path, run_id: str, experiment: str, name: str) -> None:
    run_dir = home / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment": experiment,
                "name": name,
                "tokens_seen": 1000,
                "max_tokens": 2000,
                "gpu_seconds": 30,
                "gpu_hours_remaining": 1.5,
                "best_val_loss": 1.2,
                "last_checkpoint": {},
                "complete": False,
                "updated": "2026-01-01T00:00:00+00:00",
            }
        )
    )


def test_runs_list_can_filter_by_project(slm_home: Path) -> None:
    _write_status(slm_home, "run-abc123456789", "abc_music/micro_v1", "abc run")
    _write_status(slm_home, "run-chess123456", "chess/nano", "chess run")

    all_runs = CliRunner().invoke(app, ["runs", "list"])
    filtered = CliRunner().invoke(app, ["runs", "list", "--project", "abc_music"])

    assert all_runs.exit_code == 0, all_runs.output
    assert filtered.exit_code == 0, filtered.output
    assert "abc run" in all_runs.output and "chess run" in all_runs.output
    assert "abc run" in filtered.output and "chess run" not in filtered.output


def test_runs_list_json_preserves_status_fields(slm_home: Path) -> None:
    _write_status(slm_home, "run-abc123456789", "abc_music/micro_v1", "abc run")
    _write_status(slm_home, "run-chess123456", "chess/nano", "chess run")

    result = CliRunner().invoke(app, ["runs", "list", "--json"])

    assert result.exit_code == 0, result.output
    statuses = json.loads(result.output)
    expected = [json.loads(path.read_text()) for path in (slm_home / "runs").glob("*/status.json")]
    assert sorted(statuses, key=lambda st: st["run_id"]) == sorted(
        expected, key=lambda st: st["run_id"]
    )


def test_runs_list_json_can_filter_by_project(slm_home: Path) -> None:
    _write_status(slm_home, "run-abc123456789", "abc_music/micro_v1", "abc run")
    _write_status(slm_home, "run-chess123456", "chess/nano", "chess run")

    result = CliRunner().invoke(app, ["runs", "list", "--json", "--project", "abc_music"])

    assert result.exit_code == 0, result.output
    expected = json.loads((slm_home / "runs" / "run-abc123456789" / "status.json").read_text())
    assert json.loads(result.output) == [expected]


@pytest.mark.parametrize(
    "project, has_runs", [(None, False), ("abc_music", False), ("abc_music", True)]
)
def test_runs_list_json_empty_results(slm_home: Path, project: str | None, has_runs: bool) -> None:
    if has_runs:
        _write_status(slm_home, "run-chess123456", "chess/nano", "chess run")
    args = ["runs", "list", "--json"]
    if project is not None:
        args += ["--project", project]

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "[]"
