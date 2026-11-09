# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import json
import subprocess
from contextlib import closing

import pytest

from blastradius.artifacts import Artifacts
from blastradius.cli import _build_parser, _cmd_impact
from blastradius.mcp_server import _call_get_impact, _call_temporal_impact
from blastradius.store import Store


@pytest.fixture
def graph(tmp_path):
    dependents = ["tests/test_core.py", *[f"client{i}.py" for i in range(9)]]
    data = {
        "nodes": [{"id": path} for path in ["core.py", *dependents]],
        "links": [{"source": path, "target": "core.py"} for path in dependents],
    }
    (tmp_path / "blastradius.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path, data


def test_cli_text_markdown_thresholds_and_json(graph, capsys, tmp_path):
    repo, _ = graph
    common = ["impact", "core.py", "--index", str(repo / "blastradius.json")]
    _cmd_impact(_build_parser().parse_args(common))
    default = capsys.readouterr().out
    assert "Mitigation suggestions" in default
    assert "downstream test candidates (1): `tests/test_core.py`" in default
    _cmd_impact(_build_parser().parse_args([*common, "--guidance-threshold", "11"]))
    assert "Mitigation suggestions" not in capsys.readouterr().out
    report = tmp_path / "impact.md"
    _cmd_impact(_build_parser().parse_args([*common, "--out", str(report)]))
    assert "## Mitigation suggestions" in report.read_text()
    _cmd_impact(
        _build_parser().parse_args(
            [*common, "--out", str(report), "--guidance-threshold", "11"]
        )
    )
    assert "Mitigation suggestions" not in report.read_text()
    capsys.readouterr()
    results = []
    for threshold in (0, 100):
        _cmd_impact(
            _build_parser().parse_args(
                [*common, "--json", "--guidance-threshold", str(threshold)]
            )
        )
        results.append(json.loads(capsys.readouterr().out))
    assert results[0] == results[1]
    assert set(results[0]) == {
        "file",
        "blast_score",
        "direct_dependents",
        "transitive_dependents",
        "direct_ids",
        "transitive_ids",
    }
    assert results[0]["blast_score"] == 10


def test_current_mcp_reports_share_threshold_without_new_fields(graph):
    repo, data = graph
    with closing(Store(repo / ".blastradius/index.db")) as store:
        store.sync(data)
    for handler, params in [
        (_call_get_impact, {"file_path": "core.py"}),
        (_call_temporal_impact, {"file": "core.py"}),
    ]:
        with Artifacts(repo) as artifacts:
            default = handler(params, artifacts)
            suppressed = handler({**params, "guidance_threshold": 11}, artifacts)
        assert "Mitigation suggestions" in default["report"]
        assert "Mitigation suggestions" not in suppressed["report"]
        assert "guidance_threshold" not in default
        assert {key: val for key, val in default.items() if key != "report"} == {
            key: val for key, val in suppressed.items() if key != "report"
        }


def test_historical_cli_text_and_metrics_only_mcp(graph, monkeypatch, capsys):
    repo, data = graph
    for args in (
        ("init",),
        (
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-m",
            "baseline",
        ),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with closing(Store(repo / ".blastradius/index.db")) as store:
        store.sync(data, commit=revision, snapshot=True)
    monkeypatch.chdir(repo)
    common = ["impact", "core.py", "--as-of", revision]
    _cmd_impact(_build_parser().parse_args(common))
    report = capsys.readouterr().out
    assert "[as-of " in report
    assert "downstream test candidates (1): `tests/test_core.py`" in report
    _cmd_impact(_build_parser().parse_args([*common, "--guidance-threshold", "11"]))
    assert "Mitigation suggestions" not in capsys.readouterr().out
    results = []
    for threshold in (0, 100):
        _cmd_impact(
            _build_parser().parse_args(
                [*common, "--json", "--guidance-threshold", str(threshold)]
            )
        )
        cli_result = json.loads(capsys.readouterr().out)
        with Artifacts(repo) as artifacts:
            result = _call_temporal_impact(
                {"file": "core.py", "as_of": revision, "guidance_threshold": threshold},
                artifacts,
            )
        assert result == cli_result
        results.append(result)
    assert results[0] == results[1]
    assert "report" not in results[0]
    assert set(results[0]) == {
        "file",
        "as_of",
        "blast_score",
        "direct_dependents",
        "transitive_dependents",
        "direct_ids",
        "transitive_ids",
    }
