# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from blastradius import review
from blastradius.index import CLEAN_INDEX_META_KEY, build, db_path_for
from blastradius.store import Store
from blastradius.store import db as store_db
from blastradius.temporal.history import materialize_commit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _cli(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "blastradius.cli", *args],
        cwd=repo,
        capture_output=True,
        check=check,
    )


def _symlink(path: Path, target: str) -> None:
    try:
        path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Source symlinks are unavailable: {exc}")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "config", "tag.gpgsign", "false")
    (tmp_path / ".gitignore").write_text(".blastradius/\nblastradius.json\n")
    return tmp_path


def test_review_matches_golden_from_nested_directory(repo: Path) -> None:
    (repo / "b.py").write_text("import a\n")
    (repo / "a.py").write_text("import b\n")
    (repo / "independent.py").write_text("import os\n")
    build(str(repo))
    db_path_for(repo).unlink()

    nested = repo / "nested"
    nested.mkdir()
    golden = Path(__file__).with_name("fixtures") / "review_snapshot.md"
    expected = golden.read_text(encoding="utf-8").encode("utf-8")
    assert _cli(nested, "review").stdout == expected

    destination = nested / "review.md"
    result = _cli(nested, "review", "--out", "review.md")
    assert result.stdout == b""
    assert destination.read_bytes() == expected


def test_review_ref_is_identical_across_base_snapshot_states(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "Old.java").write_text("package example;\nclass Old {}\n")
    (repo / "Main.java").write_text(
        "package example;\nimport example.Old;\nclass Main {}\n"
    )
    (repo / "café.md").write_text("Before\n")
    base = _commit(repo, "base")
    _git(repo, "tag", "-a", "baseline", "-m", "baseline")

    (repo / "Old.java").unlink()
    (repo / "New.java").write_text("package example;\nclass New {}\n")
    (repo / "Main.java").write_text(
        "package example;\nimport example.New;\nclass Main {}\n"
    )
    (repo / "café.md").write_text("After\n")
    head = _commit(repo, "swap dependency")
    build(str(repo))
    result = _cli(repo, "review", "--ref", "baseline").stdout
    report = result.decode("utf-8")
    assert f"- Base: `{base}`\n- HEAD: `{head}`\n" in report
    assert "### Added files (1)\n\n- `New.java`" in report
    assert "### Removed files (1)\n\n- `Old.java`" in report
    assert "### Modified files (2)\n\n- `Main.java`\n- `café.md`" in report
    assert "### Added edges (1)\n\n- `Main.java` → `New.java`" in report
    assert "### Removed edges (1)\n\n- `Main.java` → `Old.java`" in report

    _cli(repo, "history")
    assert _cli(repo, "review", "--ref", base).stdout == result

    with monkeypatch.context() as legacy, closing(Store(db_path_for(repo))) as store:
        legacy.setattr(store_db, "FULL_INDEX_SNAPSHOT_VERSION", "3")
        store.record_graph_snapshot(base, {"Main.java"}, set(), coverage="index")
    assert _cli(repo, "review", "--ref", base).stdout == result

    _git(repo, "checkout", "--detach", base)
    build(str(repo))
    _git(repo, "checkout", "--detach", head)
    build(str(repo))

    # These edits happened after the indexed commit and must not enter the report.
    (repo / "Main.java").write_text("class Main {}\n")
    (repo / "Uncommitted.java").write_text("class Uncommitted {}\n")

    def unexpected_analysis(*args):
        pytest.fail("Current full snapshots should be reused")

    monkeypatch.setattr(review, "analyze_commit", unexpected_analysis)
    assert review.build_review(repo, ref=base) == report
    assert _git(repo, "rev-parse", "HEAD") == head


def test_review_go_file_changes_touch_the_root_package(repo: Path) -> None:
    (repo / "go.mod").write_text("module example.com/app\n")
    (repo / "main.go").write_text("package main\n")
    (repo / "lib").mkdir()
    (repo / "lib" / "lib.go").write_text("package lib\n")
    base = _commit(repo, "base")
    build(str(repo))

    (repo / "main.go").write_text('package main\nimport "example.com/app/lib"\n')
    _commit(repo, "import library from root package")
    build(str(repo))

    report = _cli(repo, "review", "--ref", base).stdout.decode("utf-8")
    assert "## Top blast scores\n\nNo indexed files." in report
    assert "### Modified files (1)\n\n- `main.go`" in report
    assert "### Added edges (1)\n\n- `.` → `lib`" in report


@pytest.mark.parametrize("location", ["service", ".hidden", "build"])
def test_clean_index_is_independent_of_checkout_location(
    repo: Path, location: str
) -> None:
    (repo / "Main.java").write_text(
        "package example;\nimport example.Lib;\nclass Main {}\n"
    )
    (repo / "Lib.java").write_text("package example;\nclass Lib {}\n")
    (repo / "vendor").mkdir()
    (repo / "vendor" / "Ignored.java").write_text("class Ignored {}\n")
    head = _commit(repo, "base")

    checkout = repo.parent / f"{repo.name}-checkout" / location
    shutil.copytree(repo, checkout)
    expected = build(str(repo))
    data = build(str(checkout))
    assert data["nodes"] == expected["nodes"]
    assert data["links"] == expected["links"]
    assert {node["id"] for node in data["nodes"]} == {"Main.java", "Lib.java"}
    assert (
        _cli(checkout, "review", "--ref", head).stdout
        == _cli(repo, "review", "--ref", head).stdout
    )


def test_review_ref_accepts_harmless_ignored_artifacts(repo: Path) -> None:
    (repo / ".gitignore").write_text(
        ".blastradius/\nblastradius.json\n.DS_Store\n*.egg-info/\n"
    )
    (repo / "main.py").write_text("VALUE = 1\n")
    head = _commit(repo, "base")
    (repo / ".DS_Store").write_bytes(b"metadata")
    egg_info = repo / "demo.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("Name: demo\n")
    assert _git(repo, "status", "--porcelain") == ""

    build(str(repo))

    report = _cli(repo, "review", "--ref", head).stdout.decode("utf-8")
    assert "Graph coverage: full analyzer" in report
    assert "`main.py`" in report


def test_ignored_router_override_cannot_certify_a_working_graph(repo: Path) -> None:
    (repo / ".gitignore").write_text(".blastradius/\nblastradius.json\nbuild/\n")
    (repo / "auth.py").write_text(
        'router = APIRouter(prefix="/expected")\n'
        '@router.get("/users")\n'
        "def users():\n    return []\n"
    )
    (repo / "client.js").write_text('fetch("/expected/users");\n')
    head = _commit(repo, "committed API dependency")
    generated = repo / "build" / "local.py"
    generated.parent.mkdir()
    override = 'app.include_router(auth.router, prefix="/other")\n'
    edge = ("client.js", "auth.py")

    generated.write_text(override)
    assert _git(repo, "status", "--porcelain") == ""
    data = build(str(repo))
    assert edge not in {(link["source"], link["target"]) for link in data["links"]}
    with closing(Store(db_path_for(repo))) as store:
        assert store.snapshot_graph(head) is None
    rejected = _cli(repo, "review", "--ref", head, check=False)
    assert rejected.returncode == 1
    assert b"ignored files that affect the dependency graph" in rejected.stderr
    assert not rejected.stdout
    assert b"auth.py" in _cli(repo, "review").stdout

    generated.unlink()
    build(str(repo))
    assert (
        b"Graph coverage: full analyzer" in _cli(repo, "review", "--ref", head).stdout
    )
    with closing(Store(db_path_for(repo))) as store:
        verified = store.snapshot_graph(head)
        assert verified is not None and edge in verified.edges

    generated.write_text(override)
    build(str(repo))
    with closing(Store(db_path_for(repo))) as store:
        assert store.snapshot_graph(head) == verified
    assert _cli(repo, "review", "--ref", head, check=False).returncode == 1


def test_review_ref_keeps_tracked_symlinks_in_stored_deltas(repo: Path) -> None:
    (repo / "main.py").write_text("VALUE = 1\n")
    (repo / "target.py").write_text("VALUE = 2\n")
    base = _commit(repo, "base")
    build(str(repo))

    _symlink(repo / "alias-target", "target.py")
    _symlink(repo / "alias.py", "alias-target")
    (repo / "main.py").write_text("from alias import VALUE\n")
    _commit(repo, "import through tracked symlink")
    build(str(repo))

    report = _cli(repo, "review", "--ref", base).stdout.decode("utf-8")
    assert "Graph coverage: full analyzer" in report
    assert "### Added files (1)\n\n- `alias.py`" in report
    assert "### Added edges (1)\n\n- `main.py` → `alias.py`" in report


def test_review_ref_keeps_edges_for_a_git_type_change(repo: Path) -> None:
    (repo / "app.py").write_text("import before\n")
    (repo / "before.py").write_text("VALUE = 1\n")
    (repo / "after.py").write_text("VALUE = 2\n")
    (repo / "target.py").write_text("import after\n")
    base = _commit(repo, "base")
    build(str(repo))

    (repo / "app.py").unlink()
    _symlink(repo / "app.py", "target.py")
    _commit(repo, "replace app with a symlink")
    build(str(repo))

    report = _cli(repo, "review", "--ref", base).stdout.decode("utf-8")
    assert "### Modified files (1)\n\n- `app.py`" in report
    assert "### Added edges (1)\n\n- `app.py` → `after.py`" in report
    assert "### Removed edges (1)\n\n- `app.py` → `before.py`" in report
    changes = json.loads(_cli(repo, "changed-since", base, "--json").stdout)
    assert "app.py" not in changes["modified_files"]


def test_materialization_omits_escaping_and_cyclic_symlinks(repo: Path) -> None:
    (repo / "main.py").write_text("VALUE = 1\n")
    (repo / "readme.txt").write_text("Tracked documentation\n")
    outside = repo.parent / f"{repo.name}-external.txt"
    outside.write_text("External documentation\n")
    _symlink(repo / "absolute-link", str(outside))
    _symlink(repo / "escape-link", f"../{outside.name}")
    _symlink(repo / "escape-chain", "escape-link")
    _symlink(repo / "loop-a", "loop-b")
    _symlink(repo / "loop-b", "loop-a")
    _symlink(repo / "readme-link", "readme.txt")
    _symlink(repo / "readme-chain", "readme-link")
    head = _commit(repo, "base")

    destination = repo.parent / f"{repo.name}-materialized"
    destination.mkdir()
    materialize_commit(repo, head, destination)
    assert {path.name for path in destination.iterdir()} == {
        ".gitignore",
        "main.py",
        "readme.txt",
        "readme-link",
        "readme-chain",
    }
    assert (destination / "readme-chain").read_text() == "Tracked documentation\n"
    build(str(repo))
    assert _cli(repo, "review", "--ref", head).stdout


def test_graph_changing_external_symlink_prevents_certification(repo: Path) -> None:
    (repo / "main.py").write_text("VALUE = 1\n")
    outside = repo.parent / f"{repo.name}-external.css"
    outside.write_text("body { color: red; }\n")
    _symlink(repo / "external.css", str(outside))
    head = _commit(repo, "base")

    data = build(str(repo))
    assert "external.css" in {node["id"] for node in data["nodes"]}
    with closing(Store(db_path_for(repo))) as store:
        assert store.snapshot_graph(head) is None
    assert _cli(repo, "review", "--ref", head, check=False).returncode == 1


@pytest.mark.skipif(os.name == "nt", reason="Windows filenames cannot contain CR")
def test_review_ref_preserves_carriage_returns_in_tracked_paths(repo: Path) -> None:
    source = repo / "old\rname.py"
    source.write_text("VALUE = 1\n")
    base = _commit(repo, "base")
    build(str(repo))
    source.unlink()
    _commit(repo, "remove file")
    build(str(repo))

    report = _cli(repo, "review", "--ref", base).stdout.decode("utf-8")
    assert "Graph coverage: full analyzer" in report
    assert "### Removed files (1)\n\n- `old\\rname.py`" in report


def test_review_ref_requires_a_new_stamp_after_upgrade(repo: Path) -> None:
    (repo / "main.py").write_text("VALUE = 1\n")
    head = _commit(repo, "base")
    build(str(repo))
    graph = (repo / "blastradius.json").read_text(encoding="utf-8")
    old_stamp = f"v2:{head}:{hashlib.sha256(graph.encode('utf-8')).hexdigest()}"
    with closing(Store(db_path_for(repo))) as store:
        store.set_meta(CLEAN_INDEX_META_KEY, old_stamp)
        store._conn.commit()

    rejected = _cli(repo, "review", "--ref", head, check=False)
    assert rejected.returncode == 1
    assert b"run blastradius analyze . on a clean checkout" in rejected.stderr
    build(str(repo))
    assert _cli(repo, "review", "--ref", head).stdout


def test_review_ref_rejects_stale_and_dirty_indexes(repo: Path) -> None:
    source = repo / "a.py"
    source.write_text("VALUE = 1\n")
    _commit(repo, "base")
    build(str(repo))

    source.write_text("VALUE = 2\n")
    _commit(repo, "new head with identical dependency graph")
    stale = _cli(repo, "review", "--ref", "HEAD", check=False)
    assert stale.returncode == 1
    assert b"run blastradius analyze . on a clean checkout" in stale.stderr
    assert not stale.stdout

    build(str(repo))
    (repo / "dirty.py").write_text("import a\n")
    build(str(repo))
    dirty = _cli(repo, "review", "--ref", "HEAD", check=False)
    assert dirty.returncode == 1
    assert b"run blastradius analyze . on a clean checkout" in dirty.stderr
    assert not dirty.stdout
    assert b"dirty.py" in _cli(repo, "review").stdout


def test_review_ref_rejects_mismatched_artifact_and_invalid_ref(repo: Path) -> None:
    (repo / "a.py").write_text("VALUE = 1\n")
    _commit(repo, "base")
    build(str(repo))
    invalid = _cli(repo, "review", "--ref", "HEAD:a.py", check=False)
    assert invalid.returncode == 1
    assert b"Could not resolve commit ref" in invalid.stderr

    index = repo / "blastradius.json"
    data = json.loads(index.read_text(encoding="utf-8"))
    data["nodes"][0]["blast_score"] = 42
    index.write_text(json.dumps(data), encoding="utf-8")
    mismatch = _cli(repo, "review", "--ref", "HEAD", check=False)
    assert mismatch.returncode == 1
    assert b"run blastradius analyze . on a clean checkout" in mismatch.stderr
    assert not mismatch.stdout


def test_review_ref_requires_an_index_of_the_git_root(repo: Path) -> None:
    nested = repo / "nested"
    nested.mkdir()
    (nested / "a.py").write_text("VALUE = 1\n")
    _commit(repo, "base")
    build(str(nested))

    result = _cli(nested, "review", "--ref", "HEAD", check=False)
    assert result.returncode == 1
    assert b"from the Git repository root" in result.stderr
    assert b"a.py" in _cli(nested, "review").stdout


def test_review_limits_ranking_and_handles_deep_cycle(tmp_path: Path) -> None:
    chain = [f"{i:04}.py" for i in range(1200)]
    nodes = [{"id": path, "blast_score": 1} for path in reversed(chain)]
    nodes.append({"id": "self.py", "blast_score": 0})
    links = [
        {"source": source, "target": target}
        for source, target in zip(chain, chain[1:] + chain[:1])
    ]
    links.append({"source": "self.py", "target": "self.py"})
    (tmp_path / "blastradius.json").write_text(
        json.dumps({"nodes": nodes, "links": links})
    )
    report = _cli(tmp_path, "review").stdout.decode("utf-8")
    top, cycles = report.split("## Dependency cycle groups (2)")
    assert "1. `0000.py`" in top
    assert "20. `0019.py`" in top
    assert "0020.py" not in top
    assert "- `0000.py`, `0001.py`" in cycles
    assert "`1199.py`\n- `self.py`" in cycles


def test_review_reports_an_indexed_prisma_cycle(repo: Path) -> None:
    (repo / "schema.prisma").write_text(
        "model Alpha {\n  next Beta\n}\n"
        "model Beta {\n  next Gamma\n}\n"
        "model Gamma {\n  next Alpha\n}\n"
    )
    build(str(repo))

    report = _cli(repo, "review").stdout.decode("utf-8")
    assert "## Top blast scores\n\nNo indexed files." in report
    assert "## Dependency cycle groups (1)" in report
    assert (
        "- `schema.prisma:Alpha`, `schema.prisma:Beta`, `schema.prisma:Gamma`" in report
    )
