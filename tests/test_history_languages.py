# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import json
from collections import Counter
from contextlib import closing

import pytest

from blastradius import import_graph
from blastradius.artifacts import Artifacts
from blastradius.cli import _build_parser, _cmd_impact
from blastradius.import_graph import ImportGraph, Snapshot
from blastradius.index import build, db_path_for
from blastradius.mcp_server import _call_temporal_impact
from blastradius.store import Store
from blastradius.temporal import history
from tests.test_history_snapshots import commit, dependents, git, snapshot_edges, write


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    # A merge may spawn maintenance that removes object directories while the
    # normal index's language detection walks this temporary repository.
    git(tmp_path, "config", "maintenance.auto", "false")
    git(tmp_path, "config", "gc.auto", "0")
    write(tmp_path, ".gitignore", ".blastradius/\nblastradius.json\nsymbolindex.json\n")
    return tmp_path


def assert_interfaces(repo, target, revision, expected, monkeypatch, capsys):
    monkeypatch.chdir(repo)
    capsys.readouterr()
    _cmd_impact(
        _build_parser().parse_args(["impact", target, "--as-of", revision, "--json"])
    )
    cli = json.loads(capsys.readouterr().out)
    with Artifacts(repo) as artifacts:
        mcp = _call_temporal_impact({"file": target, "as_of": revision}, artifacts)
    assert set(cli["direct_ids"]) == expected
    assert cli == mcp
    assert set(cli) == {
        "file",
        "as_of",
        "blast_score",
        "direct_dependents",
        "transitive_dependents",
        "direct_ids",
        "transitive_ids",
    }


def test_go_workspace_history_head_only_and_dirty_worktree(repo, monkeypatch, capsys):
    write(
        repo,
        "app/go.mod",
        "module example.com/app\nrequire example.com/shared v0.0.0\n",
    )
    write(repo, "app/main.go", 'package main\nimport "example.com/shared/client"\n')
    write(repo, "one/go.mod", "module example.com/shared\n")
    write(repo, "one/client/lib.go", "package client\n")
    write(repo, "two/go.mod", "module example.com/two\n")
    write(repo, "two/client/lib.go", "package client\n")
    standalone = "go 1.22\nuse ./app\n"
    members = "go 1.22\nuse (\n ./app\n ./one\n)\n"
    revisions = []
    for message, config, target in [
        ("nonmember", standalone, None),
        ("add workspace member", members, "one/client"),
        (
            "redirect replacement",
            members + "replace example.com/shared => ./two\n",
            "two/client",
        ),
        ("remove membership and replacement", standalone, None),
        ("restore membership", members, "one/client"),
    ]:
        write(repo, "go.work", config)
        revisions.append((commit(repo, message), target))
    (repo / "one/client/lib.go").unlink()
    revisions.append((commit(repo, "delete target"), None))
    write(repo, "one/client/lib.go", "package client\n")
    revisions.append((commit(repo, "recreate target"), "one/client"))
    build(str(repo))
    # Historical resolution must ignore these dirty manifests.
    write(repo, "go.work", standalone + "replace example.com/shared => ./wrong\n")
    write(repo, "one/go.mod", "module example.com/dirty\n")
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in repo.rglob("*")
        if p.is_file() and ".git" not in p.parts and ".blastradius" not in p.parts
    }
    original_parse = history.parse
    parsed = Counter()
    batches = []
    original_batch = history._git_cat_file_batch
    original_run = history._run

    def record_parse(path, source):
        parsed[path] += 1
        return original_parse(path, source)

    def record_batch(root, hashes):
        batches.append(hashes)
        return original_batch(root, hashes)

    def guarded_run(command, *args, **kwargs):
        assert command[1] not in {"checkout", "switch", "restore", "show"}
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(history, "parse", record_parse)
    monkeypatch.setattr(history, "_git_cat_file_batch", record_batch)
    monkeypatch.setattr(history, "_run", guarded_run)
    with closing(Store(db_path_for(repo))) as store:
        active = store.status()["active_edges"]
        assert history.backfill(repo, store)[0] == len(revisions)
        for revision, target in revisions:
            actual = {e for e in snapshot_edges(store, revision) if e[0] == "app"}
            assert actual == ({("app", target)} if target else set())
        assert dependents(store, "one/client", revisions[1][0]) == {"app"}
        assert store.status()["active_edges"] == active
        paths = set(store.snapshot_paths(revisions[0][0]))
        assert "app" in paths and "app/main.go" not in paths
        first = {revision: snapshot_edges(store, revision) for revision, _ in revisions}
        history.backfill(repo, store, max_commits=3)
        assert first == {
            revision: snapshot_edges(store, revision) for revision, _ in revisions
        }
    # Two runs; the immutable importer is parsed only once during each run.
    assert parsed["app/main.go"] == 2
    assert any(len(batch) > 1 for batch in batches)
    for path, state in before.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == state
    assert git(repo, "rev-parse", "HEAD") == revisions[-1][0]
    assert git(repo, "branch", "--show-current") == "main"
    assert_interfaces(repo, "one/client", revisions[1][0], {"app"}, monkeypatch, capsys)


def test_go_module_replacements_and_module_names_without_importer_edits(repo):
    write(repo, "app/main.go", 'package main\nimport "example.com/shared/client"\n')
    for name in ("one", "two"):
        write(repo, f"{name}/go.mod", f"module example.com/{name}\n")
        write(repo, f"{name}/client/lib.go", "package client\n")
    manifest = "module example.com/app\nrequire example.com/shared v0.0.0\n"
    revisions = []
    for replacement in ("one", "two", None, "one"):
        write(
            repo,
            "app/go.mod",
            manifest
            + (
                f"replace example.com/shared => ../{replacement}\n"
                if replacement
                else ""
            ),
        )
        revisions.append((commit(repo, f"replace with {replacement}"), replacement))
    # A workspace member's module name is part of historical resolution context.
    write(repo, "go.work", "go 1.22\nuse (\n ./app\n ./one\n)\n")
    write(repo, "app/go.mod", manifest)
    write(repo, "one/go.mod", "module example.com/shared\n")
    named = commit(repo, "module name matches")
    write(repo, "one/go.mod", "module example.com/renamed\n")
    renamed = commit(repo, "module name changes")
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        history.backfill(repo, store)
        for revision, target in revisions:
            assert {e for e in snapshot_edges(store, revision) if e[0] == "app"} == (
                {("app", f"{target}/client")} if target else set()
            )
        assert ("app", "one/client") in snapshot_edges(store, named)
        assert ("app", "one/client") not in snapshot_edges(store, renamed)


def test_rust_module_paths_and_conditional_variants_without_consumer_edits(
    repo, monkeypatch, capsys
):
    write(
        repo,
        "Cargo.toml",
        '[package]\nname="demo"\nversion="0.1.0"\n[lib]\npath="code/root.rs"\n',
    )
    write(repo, "code/consumer.rs", "pub use crate::platform::detail::Item;\n")
    for name in ("unix", "windows"):
        write(repo, f"code/{name}.rs", "pub mod detail;\n")
        write(repo, f"code/{name}/detail.rs", "pub struct Item;\n")
    revisions = []
    for target in ("unix", "windows", "unix"):
        write(
            repo,
            "code/root.rs",
            f'#[path="{target}.rs"] mod platform;\nmod consumer;\n',
        )
        revisions.append((commit(repo, target), {f"code/{target}/detail.rs"}))
    write(
        repo,
        "code/root.rs",
        '#[cfg(unix)]\n#[path="unix.rs"] mod platform;\n#[cfg(windows)]\n#[path="windows.rs"] mod platform;\nmod consumer;\n',
    )
    revisions.append(
        (
            commit(repo, "both static variants"),
            {"code/unix/detail.rs", "code/windows/detail.rs"},
        )
    )
    (repo / "code/windows/detail.rs").unlink()
    deleted = commit(repo, "remove nested target")
    write(repo, "code/windows/detail.rs", "pub struct Item;\n")
    restored = commit(repo, "restore nested target")
    build(str(repo))
    original_parse = history.parse
    parsed = Counter()

    def recording_parse(path, source):
        parsed[path] += 1
        return original_parse(path, source)

    monkeypatch.setattr(history, "parse", recording_parse)
    # Remove the exact historical index so HEAD parity also exercises backfill.
    with closing(Store(db_path_for(repo))) as store:
        store._conn.execute("DELETE FROM commit_snapshots")
        store._conn.commit()
        history.backfill(repo, store)
        for revision, targets in revisions:
            assert {
                b for a, b in snapshot_edges(store, revision) if a == "code/consumer.rs"
            } == targets
        assert ("code/consumer.rs", "code/windows/detail.rs") not in snapshot_edges(
            store, deleted
        )
        assert ("code/consumer.rs", "code/windows/detail.rs") in snapshot_edges(
            store, restored
        )
        assert (
            snapshot_edges(store, restored)
            == ImportGraph(Snapshot.directory(repo)).graph()[1]
        )
    assert_interfaces(
        repo,
        "code/windows/detail.rs",
        revisions[0][0],
        {"code/windows.rs"},
        monkeypatch,
        capsys,
    )
    assert_interfaces(
        repo,
        "code/windows/detail.rs",
        revisions[3][0],
        {"code/windows.rs", "code/consumer.rs"},
        monkeypatch,
        capsys,
    )


@pytest.mark.parametrize(
    "inherited", [False, True], ids=["package-dependency", "workspace-dependency"]
)
def test_rust_cargo_redirects_and_target_changes_without_importer_edits(
    repo, inherited
):
    write(repo, "app/src/lib.rs", "mod consumer;\n")
    write(repo, "app/src/consumer.rs", "use model::models::Item;\n")
    for name in ("one", "two"):
        write(
            repo,
            f"{name}/Cargo.toml",
            f'[package]\nname="{name}"\nversion="0.1.0"\n[lib]\npath="code/root.rs"\n',
        )
        write(repo, f"{name}/code/root.rs", "pub mod models;\n")
        write(repo, f"{name}/code/models.rs", "pub struct Item;\n")
        write(repo, f"{name}/other/lib.rs", "pub mod models;\n")
        write(repo, f"{name}/other/models.rs", "pub struct Item;\n")
    app = '[package]\nname="app"\nversion="0.1.0"\n'
    revisions = []
    for name in ("one", "two", None, "one"):
        spec = (
            f'model={{path="{"" if inherited else "../"}{name}",package="{name}"}}\n'
            if name
            else 'model="1"\n'
        )
        if inherited:
            write(
                repo,
                "Cargo.toml",
                '[workspace]\nmembers=["app","one","two"]\n[workspace.dependencies]\n'
                + spec,
            )
            write(
                repo, "app/Cargo.toml", app + "[dependencies]\nmodel.workspace=true\n"
            )
        else:
            write(repo, "app/Cargo.toml", app + "[dependencies]\n" + spec)
        revisions.append(
            (
                commit(repo, f"Cargo dependency {name}"),
                f"{name}/code/models.rs" if name else None,
            )
        )
    write(
        repo,
        "one/Cargo.toml",
        '[package]\nname="one"\nversion="0.1.0"\n[lib]\npath="other/lib.rs"\n',
    )
    revisions.append(
        (commit(repo, "change dependency library target"), "one/other/models.rs")
    )
    build(str(repo))
    write(repo, "app/Cargo.toml", "invalid dirty manifest")
    with closing(Store(db_path_for(repo))) as store:
        history.backfill(repo, store)
        for revision, target in revisions:
            assert {
                b
                for a, b in snapshot_edges(store, revision)
                if a == "app/src/consumer.rs"
            } == ({target} if target else set())


def test_backfill_upgrades_old_go_rust_snapshots_and_preserves_full_index(
    repo, monkeypatch
):
    write(repo, "go.mod", "module example.com/app\n")
    write(repo, "main.go", 'package main\nimport "example.com/app/lib"\n')
    write(repo, "lib/lib.go", "package lib\n")
    first = commit(repo, "Go graph")
    write(repo, "src/lib.rs", "mod value;\n")
    write(repo, "src/value.rs", "pub struct Item;\n")
    head = commit(repo, "Rust graph")
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        with monkeypatch.context() as previous:
            previous.setattr(import_graph, "EXTRACTION_VERSION", "2")
            store.record_graph_snapshot(
                first, set(), set(), coverage="python,javascript,typescript,vue"
            )
        expected_head = snapshot_edges(store, head)
        history.backfill(repo, store)
        assert snapshot_edges(store, first) == {(".", "lib")}
        assert snapshot_edges(store, head) == expected_head
        info = store._conn.execute(
            "SELECT g.coverage,g.extraction_version FROM graph_snapshots g JOIN commit_snapshots c ON c.snapshot_id=g.id WHERE c.commit_hash=?",
            (first,),
        ).fetchone()
        assert info[0] == import_graph.HISTORY_COVERAGE
        assert info[1] == "3"


@pytest.mark.parametrize("language", ["go", "rust"])
def test_language_branch_and_merge_use_exact_commit_tree(repo, language):
    if language == "go":
        write(repo, "go.mod", "module example.com/app\n")
        importer, target = "app/main.go", "lib"
        clean, imported = (
            "package main\n",
            'package main\nimport "example.com/app/lib"\n',
        )
        target_source = "lib/value.go"
        write(repo, target_source, "package lib\n")
        source_id = "app"
    else:
        write(repo, "src/lib.rs", "mod consumer;\nmod value;\n")
        importer, target = "src/consumer.rs", "src/value.rs"
        clean, imported = "// no imports\n", "use crate::value::Item;\n"
        target_source = target
        write(repo, target, "pub struct Item;\n")
        source_id = importer
    write(repo, importer, clean)
    first = commit(repo, "baseline")
    git(repo, "checkout", "-b", "feature")
    write(repo, importer, imported)
    feature = commit(repo, "feature dependency")
    git(repo, "checkout", "main")
    write(
        repo, target_source, (repo / target_source).read_text() + "// changed on main\n"
    )
    main = commit(repo, "main without feature dependency")
    git(repo, "merge", "--no-ff", "feature", "-m", "merge feature")
    merged = git(repo, "rev-parse", "HEAD")
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        history.backfill(repo, store)
        assert (source_id, target) not in snapshot_edges(store, first)
        assert (source_id, target) in snapshot_edges(store, feature)
        assert (source_id, target) not in snapshot_edges(store, main)
        assert (source_id, target) in snapshot_edges(store, merged)
