# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
"""Phase 2 acceptance tests.

Acceptance criteria (from CKG-DESIGN-001):
  1. On a synthetic repo with a scripted history (add dep → remove dep),
     `impact FILE --as-of <old>` differs correctly from HEAD.
  2. `changed_since <ref>` lists the exact added/removed edges.
  3. Full backfill completes without modifying the working tree;
     commits table is populated.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from blastradius.cli import _build_parser, _cmd_impact

FIXTURE_SRC = Path(__file__).parent / "fixtures" / "simple_python"


# ── helpers ──────────────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return r.stdout.strip()


def _setup_git(repo: Path) -> None:
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@test.com")
    _git(repo, "config", "user.name", "Test")


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _run_analyze(repo: Path) -> None:
    import importlib

    import blastradius.index as idx_mod

    importlib.reload(idx_mod)
    idx_mod.build(str(repo))


# ── Fixture: add-dep then remove-dep history ─────────────────────────────────


def _make_temporal_repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    """Create repo: C1=baseline, C2=add edge B→A, C3=remove edge.

    Returns (repo, c1_hash, c2_hash, c3_hash).
    Runs analyze() after each commit to populate temporal data.
    """
    repo = tmp_path / "temporal_repo"
    repo.mkdir()
    _setup_git(repo)

    # C1: two files, no inter-file import
    (repo / "a.py").write_text('"""Module A."""\n\nX = 1\n')
    (repo / "b.py").write_text('"""Module B."""\n\nY = 2\n')
    c1 = _commit_all(repo, "C1: initial")
    _run_analyze(repo)

    # C2: a.py now imports b.py
    (repo / "a.py").write_text('"""Module A."""\nfrom b import Y\n\nX = Y + 1\n')
    c2 = _commit_all(repo, "C2: add import b in a")
    _run_analyze(repo)

    # C3: a.py no longer imports b.py
    (repo / "a.py").write_text('"""Module A."""\n\nX = 42\n')
    c3 = _commit_all(repo, "C3: remove import")
    _run_analyze(repo)

    return repo, c1, c2, c3


@pytest.fixture(scope="module")
def impact_repo(tmp_path_factory):
    return _make_temporal_repo(tmp_path_factory.mktemp("impact_cli"))


@pytest.fixture(scope="module")
def ambiguous_impact_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("ambiguous_impact_cli")
    _setup_git(repo)
    for directory in ("left", "longer"):
        (repo / directory).mkdir()
        (repo / directory / "same.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "entry.py").write_text("from left.same import VALUE\n", encoding="utf-8")
    commit = _commit_all(repo, "ambiguous file names")
    _run_analyze(repo)
    return repo, commit


@pytest.mark.parametrize("mode", ["text", "json", "report"])
def test_temporal_impact_ambiguity_closes_store(
    ambiguous_impact_repo, tmp_path, monkeypatch, capsys, mode
):
    from blastradius.store import Store

    repo, commit = ambiguous_impact_repo
    monkeypatch.chdir(repo)
    closed = []
    close = Store.close

    def record_close(store):
        close(store)
        closed.append(store)

    monkeypatch.setattr(Store, "close", record_close)
    report = tmp_path / "impact.md"
    argv = ["impact", "same.py", "--as-of", commit]
    if mode == "json":
        argv.append("--json")
    elif mode == "report":
        argv.extend(["--out", str(report)])
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        _cmd_impact(_build_parser().parse_args(argv))
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Ambiguous file same.py: left/same.py, longer/same.py; "
        "use an exact repo-relative path\n"
    )
    assert not report.exists()
    assert len(closed) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        closed[0]._conn.execute("SELECT 1")


@pytest.mark.parametrize("file_path", ["left/same.py", "./left/same.py"])
def test_temporal_impact_accepts_explicit_ambiguous_file(
    ambiguous_impact_repo, monkeypatch, capsys, file_path
):
    repo, commit = ambiguous_impact_repo
    monkeypatch.chdir(repo)
    capsys.readouterr()
    _cmd_impact(
        _build_parser().parse_args(["impact", file_path, "--as-of", commit, "--json"])
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["file"] == "left/same.py"
    assert result["direct_ids"] == ["entry.py"]
    assert captured.err == ""


@pytest.mark.parametrize("failure", ["ref", "file"])
def test_temporal_impact_early_error_closes_store(
    ambiguous_impact_repo, monkeypatch, capsys, failure
):
    from blastradius.store import Store

    repo, commit = ambiguous_impact_repo
    monkeypatch.chdir(repo)
    closed = []
    close = Store.close

    def record_close(store):
        close(store)
        closed.append(store)

    monkeypatch.setattr(Store, "close", record_close)
    file_path = "left/same.py" if failure == "ref" else "missing.py"
    ref = "no-such-ref" if failure == "ref" else commit
    capsys.readouterr()
    with pytest.raises(SystemExit) as error:
        _cmd_impact(_build_parser().parse_args(["impact", file_path, "--as-of", ref]))
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Could not resolve ref: no-such-ref\n"
        if failure == "ref"
        else "File not found in index: missing.py\n"
    )
    assert len(closed) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        closed[0]._conn.execute("SELECT 1")


@pytest.mark.parametrize("historical", [False, True], ids=["head", "historical"])
@pytest.mark.parametrize("path_kind", ["exact", "prefixed", "absolute"])
def test_impact_cli_resolves_paths(
    impact_repo, monkeypatch, capsys, historical, path_kind
):
    repo, _c1, c2, _c3 = impact_repo
    monkeypatch.chdir(repo)
    file_path = {
        "exact": "b.py",
        "prefixed": "./b.py",
        "absolute": str(repo / "b.py"),
    }[path_kind]
    argv = ["impact", file_path, "--json"]
    if historical:
        argv.extend(["--as-of", c2])
    capsys.readouterr()

    _cmd_impact(_build_parser().parse_args(argv))

    captured = capsys.readouterr()
    expected = {
        "file": "b.py",
        "blast_score": 1.0 if historical else 0.0,
        "direct_dependents": 1 if historical else 0,
        "transitive_dependents": 0,
        "direct_ids": ["a.py"] if historical else [],
        "transitive_ids": [],
    }
    if historical:
        expected["as_of"] = c2
    assert json.loads(captured.out) == expected
    assert captured.err == ""


@pytest.mark.parametrize("historical", [False, True], ids=["head", "historical"])
def test_impact_cli_text_output(impact_repo, monkeypatch, capsys, historical):
    repo, _c1, c2, _c3 = impact_repo
    monkeypatch.chdir(repo)
    argv = ["impact", "./b.py"]
    if historical:
        argv.extend(["--as-of", c2])
    capsys.readouterr()

    _cmd_impact(_build_parser().parse_args(argv))

    captured = capsys.readouterr()
    if historical:
        assert captured.out == (
            f"[as-of {c2[:12]}]  Impact: b.py\n"
            "Blast Score: 1.0  (1 direct · 0 transitive)  [CRITICAL]\n\n"
            "Direct dependents (1)\n  a.py\n\n"
            "Risk: CRITICAL — affects 1/2 files (50.0% of codebase)\n"
        )
    else:
        assert captured.out == (
            "Impact: b.py\n"
            "Blast Score: 0.0  (0 direct · 0 transitive)  [LOW]\n\n"
            "Risk: LOW — affects 0/2 files (0.0% of codebase)\n"
        )
    assert captured.err == ""


def test_impact_cli_writes_markdown(impact_repo, tmp_path, monkeypatch, capsys):
    repo, _c1, _c2, _c3 = impact_repo
    monkeypatch.chdir(repo)
    report = tmp_path / "impact.md"
    args = _build_parser().parse_args(["impact", "./b.py", "--out", str(report)])
    capsys.readouterr()

    _cmd_impact(args)

    captured = capsys.readouterr()
    assert captured.out == f"Impact report written to {report}\n"
    assert captured.err == ""
    markdown = report.read_text()
    assert markdown.startswith("# Impact Report: `b.py`\n")
    assert "| Direct dependents | 0 |" in markdown
    assert "| Transitive dependents | 0 |" in markdown
    assert "could affect **0/2** files" in markdown


@pytest.mark.parametrize("historical", [False, True], ids=["head", "historical"])
def test_impact_cli_missing_file(impact_repo, monkeypatch, capsys, historical):
    repo, _c1, c2, _c3 = impact_repo
    monkeypatch.chdir(repo)
    argv = ["impact", "missing.py"]
    if historical:
        argv.extend(["--as-of", c2])
    capsys.readouterr()

    with pytest.raises(SystemExit) as error:
        _cmd_impact(_build_parser().parse_args(argv))

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    expected = "File not found in index: missing.py\n"
    if not historical:
        expected += "Available nodes (first 20):\n  a.py\n  b.py\n"
    assert captured.err == expected


def test_impact_cli_limits_available_nodes(tmp_path, capsys):
    graph = tmp_path / "graph.json"
    graph.write_text(
        json.dumps(
            {
                "nodes": [{"id": f"module{i:02}.py"} for i in range(24, -1, -1)],
                "links": [],
            }
        )
    )
    args = _build_parser().parse_args(["impact", "missing.py", "--index", str(graph)])

    with pytest.raises(SystemExit) as error:
        _cmd_impact(args)

    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "File not found in index: missing.py",
        "Available nodes (first 20):",
        *[f"  module{i:02}.py" for i in range(20)],
    ]


def test_impact_cli_uses_explicit_index(impact_repo, tmp_path, monkeypatch, capsys):
    repo, _c1, _c2, _c3 = impact_repo
    monkeypatch.chdir(tmp_path)
    data = json.loads((repo / "blastradius.json").read_text())
    data["links"] = [{"source": "a.py", "target": "b.py", "kind": "imports"}]
    graph = tmp_path / "export.json"
    graph.write_text(json.dumps(data))
    args = _build_parser().parse_args(
        ["impact", str(repo / "b.py"), "--index", str(graph), "--json"]
    )
    capsys.readouterr()

    _cmd_impact(args)

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["file"] == "b.py"
    assert result["direct_ids"] == ["a.py"]
    assert result["direct_dependents"] == 1
    assert captured.err == ""


def test_impact_cli_resolves_historically_present_file(tmp_path, monkeypatch, capsys):
    repo, _c1, c2, _c3 = _make_temporal_repo(tmp_path)
    _git(repo, "rm", "b.py")
    _commit_all(repo, "remove b")
    _run_analyze(repo)
    monkeypatch.chdir(repo)
    args = _build_parser().parse_args(["impact", "./b.py", "--as-of", c2, "--json"])
    capsys.readouterr()

    _cmd_impact(args)

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["file"] == "b.py"
    assert result["direct_ids"] == ["a.py"]
    assert result["direct_dependents"] == 1
    assert captured.err == ""


# ── Test 1: as_of_impact differs between C2 and HEAD ─────────────────────────


def test_as_of_impact_differs_from_head(tmp_path: Path) -> None:
    """impact B --as-of C2 shows A as dependent; impact B at HEAD does not."""
    from blastradius.index import db_path_for, git_reachable
    from blastradius.store import Store

    repo, _c1, c2, c3 = _make_temporal_repo(tmp_path)
    db_path = db_path_for(repo)
    store = Store(db_path)

    # At C2: a.py imports b.py → b.py has a direct dependent
    reachable_c2 = git_reachable(repo, c2)
    blast_at_c2 = store.as_of_impact("b.py", reachable_c2)
    assert blast_at_c2 is not None, "as_of_impact returned None for C2"
    assert blast_at_c2["direct_dependents"] >= 1, (
        f"Expected a.py as dependent of b.py at C2, got: {blast_at_c2}"
    )
    assert "a.py" in blast_at_c2["direct_ids"], (
        f"a.py not in direct_ids at C2: {blast_at_c2['direct_ids']}"
    )

    # At HEAD (C3): a.py no longer imports b.py → b.py has no dependents
    reachable_c3 = git_reachable(repo, c3)
    blast_at_head = store.as_of_impact("b.py", reachable_c3)
    assert blast_at_head is not None, "as_of_impact returned None for C3"
    assert blast_at_head["direct_dependents"] == 0, (
        f"Expected 0 dependents at C3, got: {blast_at_head}"
    )

    store.close()


# ── Test 2: changed_since lists added/removed edges ──────────────────────────


def test_changed_since_edges(tmp_path: Path) -> None:
    """changed_since(reachable_C1) shows added edge at C2 and its removal."""
    from blastradius.index import db_path_for, git_reachable
    from blastradius.store import Store

    repo, c1, _c2, _c3 = _make_temporal_repo(tmp_path)
    db_path = db_path_for(repo)
    store = Store(db_path)

    # Since C1: edge a→b was added (at C2) and removed (at C3, now inactive)
    reachable_c1 = git_reachable(repo, c1)
    result = store.changed_since(reachable_c1)
    store.close()

    # The removed edge (a→b, removed at C3) should appear in removed_edges
    removed_sources = {e["source"] for e in result["removed_edges"]}
    assert "a.py" in removed_sources, (
        f"Expected a.py in removed_edges sources. Got: {result['removed_edges']}"
    )


# ── Test 3: history backfill populates commits table ─────────────────────────


def test_history_backfill_commits(tmp_path: Path) -> None:
    """blastradius history populates the commits table."""
    from blastradius.index import db_path_for
    from blastradius.store import Store
    from blastradius.temporal import backfill

    repo = tmp_path / "hist_repo"
    shutil.copytree(FIXTURE_SRC, repo)
    _setup_git(repo)

    _commit_all(repo, "init")
    (repo / "models.py").write_text((repo / "models.py").read_text() + "\n# v2\n")
    _commit_all(repo, "modify models")

    # Analyze at HEAD only (don't analyze at c1)
    _run_analyze(repo)

    db_path = db_path_for(repo)
    store = Store(db_path)
    backfill(repo, store)
    store.close()

    store2 = Store(db_path)
    commit_count = store2._conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
    store2.close()

    assert commit_count >= 2, (
        f"Expected ≥2 commits in table after backfill, got {commit_count}"
    )


# ── Test 4: history backfill sets first_seen_commit ──────────────────────────


def test_history_backfill_first_seen(tmp_path: Path) -> None:
    """After backfill, files have first_seen_commit set."""
    from blastradius.index import db_path_for
    from blastradius.store import Store
    from blastradius.temporal import backfill

    repo = tmp_path / "seen_repo"
    shutil.copytree(FIXTURE_SRC, repo)
    _setup_git(repo)
    _commit_all(repo, "init")

    _run_analyze(repo)

    db_path = db_path_for(repo)
    store = Store(db_path)
    backfill(repo, store)
    store.close()

    store2 = Store(db_path)
    null_count = store2._conn.execute(
        "SELECT COUNT(*) FROM files WHERE active=1 AND first_seen_commit IS NULL"
    ).fetchone()[0]
    store2.close()

    assert null_count == 0, (
        f"{null_count} active files still have NULL first_seen_commit after backfill"
    )


# ── Test 5: backfill never modifies working tree ─────────────────────────────


def test_history_no_working_tree_change(tmp_path: Path) -> None:
    """git status is clean before and after backfill (no checkout side-effects)."""
    from blastradius.index import db_path_for
    from blastradius.store import Store
    from blastradius.temporal import backfill

    repo = tmp_path / "clean_repo"
    shutil.copytree(FIXTURE_SRC, repo)
    _setup_git(repo)
    _commit_all(repo, "init")
    _run_analyze(repo)

    # Record file mtimes before backfill
    source_files = list(repo.rglob("*.py"))
    mtimes_before = {f: f.stat().st_mtime for f in source_files}

    db_path = db_path_for(repo)
    store = Store(db_path)
    backfill(repo, store)
    store.close()

    # Check no source file was modified
    for f in source_files:
        assert f.stat().st_mtime == mtimes_before[f], (
            f"Backfill modified working-tree file: {f}"
        )

    # Git status should show clean (only .blastradius/ and generated files differ)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "*.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert status == "", f"Working tree dirty after backfill:\n{status}"


# ── Test 6: changed-since shows added file ────────────────────────────────────


def test_changed_since_added_file(tmp_path: Path) -> None:
    """changed_since(C1) shows a file added at C2."""
    from blastradius.index import db_path_for, git_reachable
    from blastradius.store import Store

    repo = tmp_path / "addfile_repo"
    shutil.copytree(FIXTURE_SRC, repo)
    _setup_git(repo)
    c1 = _commit_all(repo, "init")
    _run_analyze(repo)

    # Add a new file and commit
    (repo / "new_module.py").write_text('"""New module."""\nVALUE = 99\n')
    _commit_all(repo, "add new_module")
    _run_analyze(repo)

    db_path = db_path_for(repo)
    store = Store(db_path)
    reachable_c1 = git_reachable(repo, c1)
    result = store.changed_since(reachable_c1)
    store.close()

    assert "new_module.py" in result["added_files"], (
        f"new_module.py not in added_files. Got: {result['added_files']}"
    )
