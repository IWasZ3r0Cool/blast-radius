# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import closing

import pytest

from blastradius.artifacts import Artifacts
from blastradius.cli import _build_parser, _cmd_impact
from blastradius.import_graph import ImportGraph, Snapshot
from blastradius.index import build, db_path_for, git_reachable
from blastradius.mcp_server import _call_temporal_impact
from blastradius.store import Store
from blastradius.temporal.history import backfill


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def write(repo, path, source):
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        file.write_bytes(source)
    else:
        file.write_text(source, encoding="utf-8")


def commit(repo, message):
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    write(tmp_path, ".gitignore", ".blastradius/\nblastradius.json\nsymbolindex.json\n")
    return tmp_path


def dependents(store, target, revision):
    result = store.as_of_impact(target, set(), commit_hash=revision)
    return None if result is None else set(result["direct_ids"])


def snapshot_edges(store, revision):
    return {
        tuple(row)
        for row in store._conn.execute(
            "SELECT e.source,e.target FROM snapshot_edges e JOIN commit_snapshots c ON c.snapshot_id=e.snapshot_id WHERE c.commit_hash=?",
            (revision,),
        )
    }


def test_head_only_index_then_history_handles_removal_reintroduction_and_deletion(
    repo, monkeypatch, capsys
):
    write(repo, "a.py", "VALUE = 1\n")
    write(repo, "b.py", "VALUE = 2\n")
    baseline = commit(repo, "baseline")
    write(repo, "a.py", "import b\n")
    added = commit(repo, "add dependency")
    write(repo, "a.py", "VALUE = 1\n")
    removed = commit(repo, "remove dependency")
    write(repo, "a.py", "import b\n")
    restored = commit(repo, "restore dependency")
    (repo / "b.py").unlink()
    deleted = commit(repo, "delete target")
    # Crucially, none of the historical commits were indexed along the way.
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        backfill(repo, store)
        assert dependents(store, "b.py", baseline) == set()
        assert dependents(store, "b.py", added) == {"a.py"}
        assert dependents(store, "b.py", removed) == set()
        assert dependents(store, "b.py", restored) == {"a.py"}
        assert dependents(store, "b.py", deleted) is None
    monkeypatch.chdir(repo)
    capsys.readouterr()
    _cmd_impact(
        _build_parser().parse_args(["impact", "b.py", "--as-of", added, "--json"])
    )
    assert json.loads(capsys.readouterr().out)["direct_ids"] == ["a.py"]
    with Artifacts(repo) as artifacts:
        result = _call_temporal_impact({"file": "b.py", "as_of": added}, artifacts)
    assert result["direct_ids"] == ["a.py"]


def test_target_additions_config_changes_and_historical_bytes(repo):
    write(repo, "main.ts", 'import "@/target.js"; import "./late";\n')
    write(repo, "tsconfig.json", '{"compilerOptions":{"paths":{"@/*":["one/*"]}}}')
    write(repo, "one/target.ts", "export const value = 1;\n")
    write(repo, "two/target.ts", "export const value = 2;\n")
    write(repo, "café.py", "VALUE = 1\n")
    write(repo, "legacy.py", b"# coding: cp1252\n# caf\xe9\nimport caf\xe9\n")
    first = commit(repo, "initial aliases and encoded source")
    write(repo, "late.ts", "export const late = 1;\n")
    second = commit(repo, "add target without changing importer")
    write(repo, "tsconfig.json", '{"compilerOptions":{"paths":{"@/*":["two/*"]}}}')
    third = commit(repo, "change configuration without changing importer")
    build(str(repo))
    # Backfill must not read the dirty working config or change its contents.
    write(repo, "tsconfig.json", '{"compilerOptions":{"paths":{"@/*":["wrong/*"]}}}')
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in repo.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".blastradius" not in path.parts
    }
    with closing(Store(db_path_for(repo))) as store:
        backfill(repo, store)
        assert ("main.ts", "one/target.ts") in snapshot_edges(store, first)
        assert ("main.ts", "late.ts") not in snapshot_edges(store, first)
        assert ("main.ts", "late.ts") in snapshot_edges(store, second)
        assert ("main.ts", "two/target.ts") in snapshot_edges(store, third)
        assert ("main.ts", "one/target.ts") not in snapshot_edges(store, third)
        assert ("legacy.py", "café.py") in snapshot_edges(store, first)
    for path, state in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == state


def test_branch_and_merge_snapshots_use_each_commits_tree(repo):
    write(repo, "a.py", "VALUE = 1\n")
    write(repo, "b.py", "VALUE = 2\n")
    baseline = commit(repo, "baseline")
    git(repo, "checkout", "-b", "feature")
    write(repo, "a.py", "import b\n")
    feature = commit(repo, "feature dependency")
    git(repo, "checkout", "main")
    write(repo, "b.py", "VALUE = 3\n")
    main = commit(repo, "independent main change")
    git(repo, "merge", "--no-ff", "feature", "-m", "merge feature")
    merge = git(repo, "rev-parse", "HEAD")
    with closing(Store(db_path_for(repo))) as store:
        backfill(repo, store)
        assert dependents(store, "b.py", baseline) == set()
        assert dependents(store, "b.py", main) == set()
        assert dependents(store, "b.py", feature) == {"a.py"}
        assert dependents(store, "b.py", merge) == {"a.py"}


def test_bounded_backfill_deduplicates_and_is_idempotent(repo):
    write(repo, "a.py", "import b\n")
    write(repo, "b.py", "VALUE = 1\n")
    old = commit(repo, "old")
    write(repo, "b.py", "VALUE = 2\n")
    middle = commit(repo, "middle")
    write(repo, "b.py", "VALUE = 3\n")
    head = commit(repo, "head")
    with closing(Store(db_path_for(repo))) as store:
        assert backfill(repo, store, max_commits=2)[0] == 2
        assert dependents(store, "b.py", old) is None
        assert dependents(store, "b.py", middle) == {"a.py"}
        assert dependents(store, "b.py", head) == {"a.py"}
        assert (
            store._conn.execute("SELECT COUNT(*) FROM graph_snapshots").fetchone()[0]
            == 1
        )
        backfill(repo, store, max_commits=2)
        assert (
            store._conn.execute("SELECT COUNT(*) FROM commit_snapshots").fetchone()[0]
            == 2
        )
        assert (
            store._conn.execute("SELECT COUNT(*) FROM graph_snapshots").fetchone()[0]
            == 1
        )


def test_uncommitted_index_cannot_overwrite_commit_snapshot(repo):
    write(repo, "a.py", "import b\n")
    write(repo, "b.py", "VALUE = 1\n")
    head = commit(repo, "initial")
    build(str(repo))
    write(repo, "a.py", "VALUE = 2\n")
    graph = build(str(repo))
    assert not graph["links"]
    with closing(Store(db_path_for(repo))) as store:
        assert dependents(store, "b.py", head) == {"a.py"}


def test_new_dirty_index_requires_backfill_for_exact_history(repo):
    write(repo, "a.py", "import b\n")
    write(repo, "b.py", "VALUE = 1\n")
    head = commit(repo, "initial")
    write(repo, "a.py", "VALUE = 2\n")
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        assert dependents(store, "b.py", head) is None
        backfill(repo, store)
        assert dependents(store, "b.py", head) == {"a.py"}


def test_git_snapshot_parity_with_working_index(repo):
    write(repo, "pkg/main.py", "from . import utils\n")
    write(repo, "pkg/__init__.py", "")
    write(repo, "pkg/utils.py", "VALUE = 1\n")
    write(repo, "main.ts", 'import "@/value";\n')
    write(repo, "tsconfig.json", '{"compilerOptions":{"paths":{"@/*":["src/*"]}}}')
    write(repo, "src/value.ts", "export const value = 1;\n")
    head = commit(repo, "fixture")
    expected_nodes, expected_edges = ImportGraph(Snapshot.directory(repo)).graph()
    with closing(Store(db_path_for(repo))) as store:
        backfill(repo, store)
        assert snapshot_edges(store, head) == expected_edges
        nodes = {
            row[0] for row in store._conn.execute("SELECT path FROM snapshot_nodes")
        }
        assert nodes == expected_nodes


def test_version_three_migration_does_not_infer_snapshots(tmp_path, monkeypatch):
    path = tmp_path / "index.db"
    with closing(Store(path)) as store:
        store.sync({"nodes": [{"id": "a.py"}], "links": []}, commit="old")
        for table in [
            "commit_snapshots",
            "snapshot_edges",
            "snapshot_nodes",
            "graph_snapshots",
        ]:
            store._conn.execute(f"DROP TABLE {table}")
        store.set_meta("schema_version", "3")
        store._conn.commit()
    connect = sqlite3.connect

    class NoFTS(sqlite3.Connection):
        def executescript(self, script):
            if "VIRTUAL TABLE" in script:
                raise sqlite3.OperationalError("no FTS5")
            return super().executescript(script)

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: connect(*args, factory=NoFTS, **kwargs),
    )
    with closing(Store(path)) as store:
        assert store.get_meta("schema_version") == "4"
        assert store.as_of_impact("a.py", {"old"}, commit_hash="old") is None
        assert store.as_of_impact("a.py", {"old"}) is not None


def test_snapshot_failure_is_atomic(tmp_path):
    with closing(Store(tmp_path / "index.db")) as store:
        store._conn.execute(
            "CREATE TRIGGER reject_edge BEFORE INSERT ON snapshot_edges BEGIN SELECT RAISE(ABORT,'test failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="test failure"):
            store.record_graph_snapshot("commit", {"a.py", "b.py"}, {("a.py", "b.py")})
        for table in [
            "graph_snapshots",
            "snapshot_nodes",
            "snapshot_edges",
            "commit_snapshots",
        ]:
            assert (
                store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
            )


def test_full_index_coverage_is_not_replaced_by_partial_backfill(tmp_path):
    with closing(Store(tmp_path / "index.db")) as store:
        store.sync(
            {
                "nodes": [{"id": "main.py"}, {"id": "rust.rs"}],
                "links": [{"source": "main.py", "target": "rust.rs"}],
            },
            commit="head",
        )
        store.record_graph_snapshot("head", {"main.py"}, set())
        assert dependents(store, "rust.rs", "head") == {"main.py"}


def test_exact_query_ignores_legacy_lifetime_metadata(repo):
    write(repo, "a.py", "import b\n")
    write(repo, "b.py", "VALUE = 1\n")
    revision = commit(repo, "initial")
    with closing(Store(db_path_for(repo))) as store:
        backfill(repo, store)
        store._conn.execute(
            "UPDATE files SET first_seen_commit=NULL, last_seen_commit=NULL, active=0"
        )
        store._conn.execute(
            "UPDATE edges SET first_seen_commit=NULL, last_seen_commit=NULL, active=0"
        )
        assert store.as_of_impact(
            "b.py", git_reachable(repo, revision), commit_hash=revision
        )["direct_ids"] == ["a.py"]


def test_file_matching_uses_requested_commit_membership(repo, monkeypatch, capsys):
    write(repo, "old/same.py", "VALUE = 1\n")
    write(repo, "main.py", "from old.same import VALUE\n")
    first = commit(repo, "one matching file")
    write(repo, "new/same.py", "VALUE = 2\n")
    commit(repo, "second matching basename")
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        backfill(repo, store)
    monkeypatch.chdir(repo)
    capsys.readouterr()
    _cmd_impact(
        _build_parser().parse_args(["impact", "same.py", "--as-of", first, "--json"])
    )
    result = json.loads(capsys.readouterr().out)
    assert result["file"] == "old/same.py"
    assert result["direct_ids"] == ["main.py"]
    with Artifacts(repo) as artifacts:
        result = _call_temporal_impact({"file": "same.py", "as_of": first}, artifacts)
    assert result["file"] == "old/same.py"


def test_dirty_go_source_does_not_create_a_commit_snapshot(repo):
    write(repo, "go.mod", "module example.com/demo\n\ngo 1.22\n")
    write(repo, "main.go", "package main\n")
    head = commit(repo, "go package")
    write(repo, "main.go", 'package main\nimport "fmt"\n')
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        assert store.snapshot_paths(head) is None
