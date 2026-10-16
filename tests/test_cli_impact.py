# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import json

import pytest

from blastradius.cli import (
    _build_parser,
    _cmd_dependencies,
    _cmd_impact,
    _resolve_cli_file_id,
)


@pytest.mark.parametrize("collection_type", [list, set], ids=["list", "set"])
@pytest.mark.parametrize(
    ("file_path", "candidate_ids", "expected"),
    [
        pytest.param(
            "service.py", ["pkg/service.py", "service.py"], "service.py", id="exact"
        ),
        pytest.param(
            "./pkg/service.py", ["pkg/service.py"], "pkg/service.py", id="dot-prefix"
        ),
        pytest.param(
            "service.py", ["pkg/service.py"], "pkg/service.py", id="short-path"
        ),
        pytest.param(
            "/missing/pkg/service.py",
            ["pkg/service.py"],
            "pkg/service.py",
            id="absolute",
        ),
        pytest.param(
            "missing/pkg/service.py",
            ["pkg/service.py"],
            "pkg/service.py",
            id="long-path",
        ),
        pytest.param("vice.py", ["service.py"], "service.py", id="partial-name"),
        pytest.param("missing.py", ["service.py"], None, id="not-found"),
        pytest.param("service.py", [], None, id="no-candidates"),
        pytest.param(
            "../../pkg/service.py",
            ["pkg/service.py"],
            "pkg/service.py",
            id="parent-prefix",
        ),
        pytest.param(
            ".config.py",
            ["pkg/config.py", ".config.py"],
            ".config.py",
            id="exact-dotfile",
        ),
        pytest.param(".config.py", ["config.py"], "config.py", id="stripped-dotfile"),
        pytest.param(
            r"pkg\service.py", ["pkg/service.py"], None, id="literal-separator"
        ),
        pytest.param(
            "pkg/./service.py", ["pkg/service.py"], None, id="literal-dot-segment"
        ),
    ],
)
def test_resolve_cli_file_id(file_path, candidate_ids, expected, collection_type):
    candidates = collection_type(candidate_ids)
    original = candidates.copy()

    assert _resolve_cli_file_id(file_path, candidates) == expected
    assert candidates == original


@pytest.mark.parametrize("file_path", ["service.py", "./service.py", "././service.py"])
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_cli_paths_take_priority(file_path, reverse):
    candidates = ["pkg/service.py", "service.py"]
    if reverse:
        candidates.reverse()
    assert _resolve_cli_file_id(file_path, candidates) == "service.py"


def test_dot_prefix_preserves_dotfile_name():
    assert (
        _resolve_cli_file_id("./.config.py", ["config.py", ".config.py"])
        == ".config.py"
    )


@pytest.mark.parametrize("collection_type", [list, set])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("other", ["z/service.py", "longer/service.py"])
def test_cli_rejects_ambiguous_suffixes(collection_type, reverse, other):
    candidates = [other, "a/service.py"]
    if reverse:
        candidates.reverse()
    candidates = collection_type(candidates)
    original = candidates.copy()
    with pytest.raises(ValueError) as error:
        _resolve_cli_file_id("service.py", candidates)
    assert str(error.value) == (
        f"Ambiguous file service.py: {', '.join(sorted(candidates))}; "
        "use an exact repo-relative path"
    )
    assert candidates == original


def test_duplicate_candidates_are_one_match():
    assert (
        _resolve_cli_file_id("service.py", ["pkg/service.py"] * 2) == "pkg/service.py"
    )


@pytest.fixture
def cli_graph(tmp_path):
    data = {
        "nodes": [
            {"id": "pkg/service.py", "imports": [], "imported_by": ["service.py"]},
            {"id": "service.py", "imports": ["pkg/service.py"], "imported_by": []},
            {"id": "a/same.py"},
            {"id": "longer/same.py"},
        ],
        "links": [
            {"source": "service.py", "target": "pkg/service.py", "kind": "imports"}
        ],
    }
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(data), encoding="utf-8")
    return graph


@pytest.mark.parametrize(
    ("command", "mode"),
    [
        ("impact", "text"),
        ("impact", "json"),
        ("impact", "report"),
        ("dependencies", "text"),
        ("dependencies", "json"),
    ],
)
def test_cli_ambiguity_has_no_output(cli_graph, tmp_path, capsys, command, mode):
    report = tmp_path / "impact.md"
    argv = [command, "same.py", "--index", str(cli_graph)]
    if mode == "json":
        argv.append("--json")
    elif mode == "report":
        argv.extend(["--out", str(report)])
    handler = _cmd_impact if command == "impact" else _cmd_dependencies
    with pytest.raises(SystemExit) as error:
        handler(_build_parser().parse_args(argv))
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Ambiguous file same.py: a/same.py, longer/same.py; "
        "use an exact repo-relative path\n"
    )
    assert not report.exists()


@pytest.mark.parametrize("command", ["impact", "dependencies"])
@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("service.py", "service.py"),
        ("./service.py", "service.py"),
        ("pkg/service.py", "pkg/service.py"),
        ("./pkg/service.py", "pkg/service.py"),
        ("onger/same.py", "longer/same.py"),
    ],
)
def test_cli_explicit_paths_and_unique_suffixes(
    cli_graph, capsys, command, file_path, expected
):
    argv = [command, file_path, "--index", str(cli_graph), "--json"]
    handler = _cmd_impact if command == "impact" else _cmd_dependencies
    handler(_build_parser().parse_args(argv))
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["file"] == expected
    if command == "dependencies":
        assert result == {
            "file": expected,
            "imports": ["pkg/service.py"] if expected == "service.py" else [],
            "imported_by": ["service.py"] if expected == "pkg/service.py" else [],
            "blast_score": 0,
        }
    assert captured.err == ""


def test_dependencies_text_output(cli_graph, capsys):
    _cmd_dependencies(
        _build_parser().parse_args(
            ["dependencies", "./service.py", "--index", str(cli_graph)]
        )
    )
    captured = capsys.readouterr()
    assert captured.out == (
        "File: service.py  (blast score: 0.0)\n"
        "\nImports (1):\n  pkg/service.py\n"
        "\nImported by (0):\n"
    )
    assert captured.err == ""


def test_dependencies_missing_file(cli_graph, capsys):
    with pytest.raises(SystemExit) as error:
        _cmd_dependencies(
            _build_parser().parse_args(
                ["dependencies", "missing.py", "--index", str(cli_graph)]
            )
        )
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "File not found in index: missing.py\n"
