# Copyright 2026 Christopher Wilson
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import copy
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from io import BytesIO

import pytest

from blastradius import import_graph
from blastradius.analyze import analyze
from blastradius.artifacts import Artifacts
from blastradius.cli import _build_parser, _cmd_impact
from blastradius.impact import compute_blast_radius
from blastradius.import_graph import ImportGraph, Snapshot
from blastradius.import_graph.parsing import javascript_imports, python_imports
from blastradius.index import build, db_path_for, git_reachable
from blastradius.mcp_server import _call_temporal_impact
from blastradius.store import Store
from blastradius.store import db as store_db
from blastradius.temporal import history
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


def test_sync_requires_snapshot_opt_in(tmp_path):
    graph = {
        "nodes": [{"id": "a.py"}, {"id": "b.py"}],
        "links": [{"source": "a.py", "target": "b.py"}],
    }
    with closing(Store(tmp_path / "index.db")) as store:
        store.sync(graph, commit="head")
        assert store.snapshot_paths("head") is None
        assert store.status()["active_edges"] == 1

        store.sync(graph, commit="head", snapshot=True)
        assert dependents(store, "b.py", "head") == {"a.py"}

        store.sync({"nodes": graph["nodes"], "links": []}, commit="head")
        assert store.status()["active_edges"] == 0
        assert dependents(store, "b.py", "head") == {"a.py"}


def test_snapshot_projection_preserves_internal_topology(tmp_path):
    graph = {
        "nodes": [
            {"id": "main.py", "type": "module"},
            {"id": "crate/lib.rs", "type": "module", "language": "rust"},
            {"id": "go/package", "type": "module", "language": "go"},
            {"id": "database", "type": "database"},
            {"id": "requests", "type": "import"},
        ],
        "links": [
            {"source": "main.py", "target": "crate/lib.rs"},
            {"source": "crate/lib.rs", "target": "go/package"},
            {"source": "go/package", "target": "database"},
            {"source": "main.py", "target": "requests"},
        ],
    }
    original = copy.deepcopy(graph)
    expected = compute_blast_radius(graph["nodes"], graph["links"])
    with closing(Store(tmp_path / "index.db")) as store:
        store.sync(graph, commit="head", snapshot=True)
        assert set(store.snapshot_paths("head")) == {
            "main.py",
            "crate/lib.rs",
            "go/package",
            "database",
        }
        assert snapshot_edges(store, "head") == {
            (link["source"], link["target"])
            for link in graph["links"]
            if link["target"] != "requests"
        }
        for path in store.snapshot_paths("head"):
            assert store.as_of_impact(path, set(), commit_hash="head") == expected[path]
        # The current graph and JSON input still include external dependencies.
        assert store.status()["active_edges"] == 4
        assert graph == original


@pytest.mark.parametrize("via_sync", [False, True], ids=["record", "sync"])
@pytest.mark.parametrize("existing", [False, True], ids=["new", "replacement"])
@pytest.mark.parametrize("edge", [("a.py", "missing.py"), ("missing.py", "a.py")])
def test_dangling_snapshot_edges_fail_before_writes(tmp_path, via_sync, existing, edge):
    with closing(Store(tmp_path / "index.db")) as store:
        if existing:
            store.sync(
                {
                    "nodes": [{"id": "a.py"}, {"id": "b.py"}],
                    "links": [{"source": "a.py", "target": "b.py"}],
                },
                commit="head",
                snapshot=True,
            )
        before = list(store._conn.iterdump())
        assert not store._conn.in_transaction
        with pytest.raises(ValueError, match="missing.py"):
            if via_sync:
                store.sync(
                    {
                        "nodes": [{"id": "a.py"}, {"id": "new.py"}],
                        "links": [{"source": edge[0], "target": edge[1]}],
                    },
                    commit="head",
                    snapshot=True,
                )
            else:
                store.record_graph_snapshot("head", {"a.py", "new.py"}, {edge})
        assert not store._conn.in_transaction
        assert list(store._conn.iterdump()) == before


def test_external_projection_cannot_hide_missing_internal_endpoint(tmp_path):
    with closing(Store(tmp_path / "index.db")) as store:
        before = list(store._conn.iterdump())
        with pytest.raises(ValueError, match="missing.py"):
            store.sync(
                {
                    "nodes": [{"id": "requests", "type": "import"}],
                    "links": [{"source": "missing.py", "target": "requests"}],
                },
                commit="head",
                snapshot=True,
            )
        assert list(store._conn.iterdump()) == before


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


def test_impact_cli_distinguishes_missing_history_from_a_failed_snapshot_read(repo):
    write(repo, "a.py", "import b\n")
    write(repo, "b.py", "VALUE = 1\n")
    baseline = commit(repo, "baseline")
    write(repo, "b.py", "VALUE = 2\n")
    head = commit(repo, "update dependency")
    build(str(repo))

    def impact_at(revision):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "blastradius.cli",
                "impact",
                "b.py",
                "--as-of",
                revision,
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )

    missing = impact_at(baseline)
    assert missing.returncode == 1
    assert not missing.stdout
    assert f"No temporal data for b.py at {baseline}" in missing.stderr
    assert "Run `blastradius history` to backfill" in missing.stderr

    with closing(sqlite3.connect(db_path_for(repo))) as connection:
        connection.execute(
            "ALTER TABLE snapshot_edges RENAME COLUMN target TO incompatible_target"
        )
        connection.commit()

    failed = impact_at(head)
    assert failed.returncode == 1
    assert not failed.stdout
    assert failed.stderr.strip() == (
        "Impact: could not read temporal index: no such column: target"
    )


def test_target_additions_config_changes_and_historical_bytes(repo, monkeypatch):
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
    parse = history.parse
    parsed_paths = []

    def recording_parse(path, source):
        parsed_paths.append(path)
        return parse(path, source)

    monkeypatch.setattr(history, "parse", recording_parse)
    with closing(Store(db_path_for(repo))) as store:
        backfill(repo, store)
        assert ("main.ts", "one/target.ts") in snapshot_edges(store, first)
        assert ("main.ts", "late.ts") not in snapshot_edges(store, first)
        assert ("main.ts", "late.ts") in snapshot_edges(store, second)
        assert ("main.ts", "two/target.ts") in snapshot_edges(store, third)
        assert ("main.ts", "one/target.ts") not in snapshot_edges(store, third)
        assert ("legacy.py", "café.py") in snapshot_edges(store, first)
    assert parsed_paths.count("main.ts") == 1
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
            snapshot=True,
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


@pytest.mark.parametrize("ignore_location", ["src/.gitignore", ".git/info/exclude"])
def test_ignored_uncommitted_source_cannot_enter_head_snapshot(repo, ignore_location):
    write(repo, "main.py", "from src.generated import VALUE\n")
    write(repo, "src/tracked.py", "VALUE = 1\n")
    write(
        repo,
        ignore_location,
        "generated.py\n"
        if ignore_location.startswith("src/")
        else "src/generated.py\n",
    )
    head = commit(repo, "tracked files and ignore rule")
    write(repo, "src/generated.py", "VALUE = 2\n")
    assert git(repo, "status", "--porcelain") == ""

    expected = analyze(str(repo))
    data = build(str(repo))
    assert {node["id"] for node in data["nodes"]} == {
        node["id"] for node in expected["nodes"]
    }
    assert data["links"] == expected["links"]
    assert ("main.py", "src/generated.py") in {
        (link["source"], link["target"]) for link in data["links"]
    }
    with closing(Store(db_path_for(repo))) as store:
        assert store.snapshot_paths(head) is None


def test_ignored_outputs_and_skipped_directories_allow_head_snapshot(repo):
    write(repo, "main.py", "VALUE = 1\n")
    write(
        repo,
        ".gitignore",
        ".blastradius/\nblastradius.json\nsymbolindex.json\nnode_modules/\n.venv/\ndist/\n",
    )
    head = commit(repo, "clean source")
    for path in [
        "node_modules/generated.py",
        ".venv/generated.py",
        "dist/generated.py",
        ".blastradius/output",
        "blastradius.json",
        "symbolindex.json",
    ]:
        write(repo, path, "ignored output\n")
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        assert store.snapshot_paths(head) == ["main.py"]


@pytest.mark.parametrize(
    "valid_config", [True, False], ids=["custom-extension", "invalid-markdown"]
)
def test_historical_extends_reads_any_advertised_file(
    repo, monkeypatch, capsys, valid_config
):
    filename = "config.base" if valid_config else "README.md"
    write(repo, "tsconfig.json", json.dumps({"extends": "./" + filename}))
    write(
        repo,
        filename,
        '{"extends":"./config.shared"}' if valid_config else "# Not JSON\n",
    )
    write(repo, "config.shared", '{"compilerOptions":{"paths":{"@/*":["src/*"]}}}')
    write(repo, "main.ts", 'import "@/target"; import "./direct";\n')
    write(repo, "src/target.ts", "export const value = 1;\n")
    write(repo, "direct.ts", "export const value = 1;\n")
    first = commit(repo, "custom extends target")
    write(repo, "direct.ts", "export const value = 2;\n")
    head = commit(repo, "unchanged configuration")
    expected_nodes, expected_edges = ImportGraph(Snapshot.directory(repo)).graph()
    requested_blobs = []
    read_batch = history._git_cat_file_batch

    def recording_batch(root, hashes):
        requested_blobs.extend(hashes)
        return read_batch(root, hashes)

    monkeypatch.setattr(history, "_git_cat_file_batch", recording_batch)
    capsys.readouterr()
    with closing(Store(db_path_for(repo))) as store:
        assert backfill(repo, store)[0] == 2
        assert set(store.snapshot_paths(head)) == expected_nodes
        for revision in [first, head]:
            assert snapshot_edges(store, revision) == expected_edges
            assert ("main.ts", "direct.ts") in snapshot_edges(store, revision)
    assert requested_blobs.count(git(repo, "rev-parse", head + ":" + filename)) == 1
    if valid_config:
        assert ("main.ts", "src/target.ts") in expected_edges
        assert (
            requested_blobs.count(git(repo, "rev-parse", head + ":config.shared")) == 1
        )
    else:
        assert "README.md: invalid configuration" in capsys.readouterr().err


def test_backfill_repairs_older_contaminated_index_snapshot(repo, monkeypatch):
    write(repo, "main.py", "VALUE = 1\n")
    head = commit(repo, "committed source")
    with closing(Store(db_path_for(repo))) as store:
        with monkeypatch.context() as previous_version:
            previous_version.setattr(store_db, "FULL_INDEX_SNAPSHOT_VERSION", "1")
            store.record_graph_snapshot(
                head,
                {"main.py", "uncommitted.py"},
                {("main.py", "uncommitted.py")},
                coverage="index",
            )
        assert "uncommitted.py" in store.snapshot_paths(head)
        backfill(repo, store)
        assert store.snapshot_paths(head) == ["main.py"]
        assert snapshot_edges(store, head) == set()
        row = store._conn.execute(
            "SELECT g.extraction_version,g.coverage FROM commit_snapshots c JOIN graph_snapshots g ON g.id=c.snapshot_id WHERE c.commit_hash=?",
            (head,),
        ).fetchone()
        assert row[0] == import_graph.EXTRACTION_VERSION
        assert row[1] != "index"


def test_backfill_preserves_full_java_snapshots_from_version_three(
    repo, monkeypatch, capsys
):
    write(repo, "Main.java", "package example;\nimport example.Lib;\nclass Main {}\n")
    write(repo, "Lib.java", "package example;\nclass Lib {}\n")
    head = commit(repo, "Java dependency")
    with monkeypatch.context() as previous_version:
        previous_version.setattr(store_db, "FULL_INDEX_SNAPSHOT_VERSION", "3")
        build(str(repo))

    with closing(Store(db_path_for(repo))) as store:
        before = store.snapshot_graph(head)
        assert before is not None and before.extraction_version == "3"
        assert dependents(store, "Lib.java", head) == {"Main.java"}
        backfill(repo, store)
        assert store.snapshot_graph(head) == before
    monkeypatch.chdir(repo)
    capsys.readouterr()
    _cmd_impact(
        _build_parser().parse_args(["impact", "Lib.java", "--as-of", head, "--json"])
    )
    assert json.loads(capsys.readouterr().out)["direct_ids"] == ["Main.java"]


def test_materialization_reads_blob_payloads_in_bounded_chunks(tmp_path):
    class BoundedStream(BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= 1024 * 1024
            return super().read(size)

    blob = "a" * 40
    payload = b"x" * (3 * 1024 * 1024)
    header = f"{blob} blob {len(payload)}\n".encode("ascii")
    tree = {"asset.bin": ("100644", blob)}
    with BoundedStream(header + payload + b"\n") as stream:
        assert history._read_tree_blobs(stream, tree, tmp_path) == {}
    asset = tmp_path / "asset.bin"
    assert asset.read_bytes() == payload
    asset.unlink()
    with (
        BoundedStream(header + payload[:-2]) as stream,
        pytest.raises(RuntimeError, match="Incomplete Git blob"),
    ):
        history._read_tree_blobs(stream, tree, tmp_path)


@pytest.mark.parametrize("extension", ["py", "ts"])
def test_backfill_parses_unchanged_sources_once_above_lru_capacity(
    repo, monkeypatch, extension
):
    for index in range(513):
        source = (
            f"VALUE = {index}\n"
            if extension == "py"
            else f"export const value = {index};\n"
        )
        write(repo, f"file{index:04}.{extension}", source)
    commit(repo, "513 distinct sources")
    git(repo, "commit", "--allow-empty", "-m", "unchanged tree one")
    git(repo, "commit", "--allow-empty", "-m", "unchanged tree two")
    write(
        repo,
        f"file0000.{extension}",
        "UPDATED = 1\n" if extension == "py" else "export const updated = true;\n",
    )
    commit(repo, "change one source")
    parser = python_imports if extension == "py" else javascript_imports
    parser.cache_clear()
    misses = []
    with closing(Store(db_path_for(repo))) as store:
        record_snapshot = store.record_graph_snapshot

        def recording_snapshot(*args, **kwargs):
            misses.append(parser.cache_info().misses)
            return record_snapshot(*args, **kwargs)

        monkeypatch.setattr(store, "record_graph_snapshot", recording_snapshot)
        assert backfill(repo, store)[0] == 4
    assert misses == [513, 513, 513, 514]


@pytest.mark.parametrize("extension", ["py", "go", "rs"])
def test_backfill_parsed_cache_discards_removed_and_excluded_files(
    repo, monkeypatch, extension
):
    kept, removed, excluded = (
        f"{name}.{extension}" for name in ("kept", "removed", "excluded")
    )
    for path in [kept, removed, excluded]:
        write(
            repo,
            path,
            {
                "py": f"# {path}\nVALUE = 1\n",
                "go": "package demo\n",
                "rs": "pub struct Item;\n",
            }[extension],
        )
    commit(repo, "initial files")
    (repo / removed).unlink()
    ignores = (repo / ".gitignore").read_text()
    write(repo, ".gitignore", ignores + excluded + "\n")
    commit(repo, "remove and exclude files")
    write(repo, ".gitignore", ignores)
    commit(repo, "include file again")

    parse = history.parse
    parsed_paths = []
    inventories = []

    def recording_parse(path, source):
        parsed_paths.append(path)
        return parse(path, source)

    def recording_graph(snapshot, *, parsed_sources):
        inventories.append(set(parsed_sources))
        return ImportGraph(snapshot, parsed_sources=parsed_sources)

    monkeypatch.setattr(history, "parse", recording_parse)
    monkeypatch.setattr(history, "ImportGraph", recording_graph)
    with closing(Store(db_path_for(repo))) as store:
        assert backfill(repo, store)[0] == 3
    assert inventories == [
        {kept, removed, excluded},
        {kept},
        {kept, excluded},
    ]
    assert parsed_paths.count(kept) == 1
    assert parsed_paths.count(removed) == 1
    assert parsed_paths.count(excluded) == 2
