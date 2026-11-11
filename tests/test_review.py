# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from blastradius import import_graph
from blastradius.index import CLEAN_INDEX_META_KEY, build, db_path_for
from blastradius.store import Store


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


@pytest.mark.parametrize("base_index", ["missing", "backfilled", "full", "old"])
def test_review_ref_includes_removed_files_and_edges(
    repo: Path, base_index: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "old.py").write_text("VALUE = 1\n")
    (repo / "main.py").write_text("import old\n")
    (repo / "café.md").write_text("Before\n")
    base = _commit(repo, "base")
    _git(repo, "tag", "-a", "baseline", "-m", "baseline")
    if base_index == "full":
        build(str(repo))
    elif base_index == "old":
        with monkeypatch.context() as legacy:
            legacy.setattr(import_graph, "EXTRACTION_VERSION", "3")
            build(str(repo))

    (repo / "old.py").unlink()
    (repo / "new.py").write_text("VALUE = 2\n")
    (repo / "main.py").write_text("import new\n")
    (repo / "café.md").write_text("After\n")
    head = _commit(repo, "swap dependency")
    build(str(repo))
    if base_index == "backfilled":
        _cli(repo, "history")

    # These edits happened after the indexed commit and must not enter the report.
    (repo / "main.py").write_text("VALUE = 3\n")
    (repo / "uncommitted.py").write_text("VALUE = 4\n")
    result = _cli(repo, "review", "--ref", "baseline")
    assert _cli(repo, "review", "--ref", base).stdout == result.stdout
    report = result.stdout.decode("utf-8")
    assert f"- Base: `{base}`\n- HEAD: `{head}`\n" in report
    assert "### Added files (1)\n\n- `new.py`" in report
    assert "### Removed files (1)\n\n- `old.py`" in report
    assert "### Modified files (2)\n\n- `café.md`\n- `main.py`" in report
    assert "### Added edges (1)\n\n- `main.py` → `new.py`" in report
    assert "### Removed edges (1)\n\n- `main.py` → `old.py`" in report
    assert "uncommitted.py" not in report
    assert ("stored index snapshots" in report) == (base_index == "full")
    assert ("historical import graphs" in report) == (base_index != "full")
    assert _git(repo, "rev-parse", "HEAD") == head


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
    assert "stored index snapshots" in report
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
    assert b"stored index snapshots" in _cli(repo, "review", "--ref", head).stdout
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

    try:
        (repo / "alias.py").symlink_to("target.py")
    except OSError as exc:
        pytest.skip(f"Source symlinks are unavailable: {exc}")
    (repo / "main.py").write_text("from alias import VALUE\n")
    _commit(repo, "import through tracked symlink")
    build(str(repo))

    report = _cli(repo, "review", "--ref", base).stdout.decode("utf-8")
    assert "stored index snapshots" in report
    assert "### Added files (1)\n\n- `alias.py`" in report
    assert "### Added edges (1)\n\n- `main.py` → `alias.py`" in report


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
    assert "stored index snapshots" in report
    assert "### Removed files (1)\n\n- `old\\rname.py`" in report


def test_review_ref_requires_a_new_stamp_after_upgrade(repo: Path) -> None:
    (repo / "main.py").write_text("VALUE = 1\n")
    head = _commit(repo, "base")
    build(str(repo))
    graph = (repo / "blastradius.json").read_text(encoding="utf-8")
    old_stamp = f"{head}:{hashlib.sha256(graph.encode('utf-8')).hexdigest()}"
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
