# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import copy
import hashlib
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest

from blastradius.store import Store


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg/service.py").write_bytes(b"local source\n")
    return repo.resolve()


def stored_hash(store, path):
    row = store._conn.execute(
        "SELECT content_hash, active FROM files WHERE path=?", (path,)
    ).fetchone()
    assert row is not None
    assert row["active"] == 1
    return row["content_hash"]


def forbid_reads(monkeypatch):
    reader = Mock(side_effect=AssertionError("Unexpected file read"))
    monkeypatch.setattr(Path, "read_bytes", reader)
    return reader


def make_symlink(link, target):
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlinks unavailable: {exc}")


@pytest.mark.parametrize("path", ["pkg/service.py", "pkg/../pkg/service.py"])
def test_hash_insert_and_update_from_another_directory(
    repository, tmp_path, monkeypatch, path
):
    monkeypatch.chdir(tmp_path)
    graph = {"nodes": [{"id": path}], "links": []}
    original = copy.deepcopy(graph)
    with closing(Store(repository / ".blastradius/index.db")) as store:
        store.set_meta("repo_root", str(repository))
        for content in (b"local source\n", b"updated source\n"):
            (repository / "pkg/service.py").write_bytes(content)
            store.sync(graph)
            assert stored_hash(store, path) == hashlib.sha256(content).hexdigest()[:16]
        assert graph == original


@pytest.mark.parametrize(
    "path_kind", ["parent", "sibling-prefix", "absolute-outside", "absolute-inside"]
)
def test_rejected_paths_are_never_read(repository, tmp_path, monkeypatch, path_kind):
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside source\n")
    sibling = tmp_path / "repo-other"
    sibling.mkdir()
    (sibling / "service.py").write_bytes(b"sibling source\n")
    path = {
        "parent": "../outside.py",
        "sibling-prefix": "../repo-other/service.py",
        "absolute-outside": str(outside),
        "absolute-inside": str(repository / "pkg/service.py"),
    }[path_kind]
    with closing(Store(repository / ".blastradius/index.db")) as store:
        store.set_meta("repo_root", str(repository))
        reader = forbid_reads(monkeypatch)
        # Also replace a previously supplied hash on the update path.
        store.sync({"nodes": [{"id": path, "content_hash": "old"}], "links": []})
        store.sync({"nodes": [{"id": path}], "links": []})
        assert stored_hash(store, path) is None
        reader.assert_not_called()


@pytest.mark.parametrize(
    "root_kind", ["missing", "empty", "relative", "absent", "file", "invalid"]
)
def test_invalid_repo_root_disables_reads(repository, tmp_path, monkeypatch, root_kind):
    monkeypatch.chdir(repository)
    values = {
        "empty": "",
        "relative": ".",
        "absent": str(tmp_path / "absent"),
        "file": str(repository / "pkg/service.py"),
        "invalid": str(tmp_path / "\0invalid"),
    }
    with closing(Store(repository / ".blastradius/index.db")) as store:
        if root_kind != "missing":
            store.set_meta("repo_root", values[root_kind])
        reader = forbid_reads(monkeypatch)
        store.sync({"nodes": [{"id": "pkg/service.py"}], "links": []})
        assert stored_hash(store, "pkg/service.py") is None
        reader.assert_not_called()


@pytest.mark.parametrize("path", ["missing.py", "pkg", "invalid\0path"])
def test_unavailable_targets_keep_graph_rows(repository, monkeypatch, path):
    with closing(Store(repository / ".blastradius/index.db")) as store:
        store.set_meta("repo_root", str(repository))
        reader = forbid_reads(monkeypatch)
        store.sync({"nodes": [{"id": path}], "links": []})
        assert stored_hash(store, path) is None
        reader.assert_not_called()


def test_unreadable_target_does_not_abort_sync(repository, monkeypatch):
    with closing(Store(repository / ".blastradius/index.db")) as store:
        store.set_meta("repo_root", str(repository))
        monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=PermissionError))
        store.sync({"nodes": [{"id": "pkg/service.py"}], "links": []})
        assert stored_hash(store, "pkg/service.py") is None


@pytest.mark.parametrize("target_kind", ["internal", "external", "loop"])
def test_symlink_hash_containment(repository, tmp_path, monkeypatch, target_kind):
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside source\n")
    link = repository / "link.py"
    target = {
        "internal": repository / "pkg/service.py",
        "external": outside,
        "loop": link,
    }[target_kind]
    make_symlink(link, target)
    with closing(Store(repository / ".blastradius/index.db")) as store:
        store.set_meta("repo_root", str(repository))
        if target_kind != "internal":
            reader = forbid_reads(monkeypatch)
        store.sync({"nodes": [{"id": "link.py"}], "links": []})
        expected = (
            hashlib.sha256(b"local source\n").hexdigest()[:16]
            if target_kind == "internal"
            else None
        )
        assert stored_hash(store, "link.py") == expected
        if target_kind != "internal":
            reader.assert_not_called()


@pytest.mark.parametrize("value", ["supplied-hash", None])
def test_supplied_hash_does_not_read_files(repository, monkeypatch, value):
    graph = {"nodes": [{"id": "pkg/service.py", "content_hash": value}], "links": []}
    original = copy.deepcopy(graph)
    with closing(Store(repository / ".blastradius/index.db")) as store:
        store.set_meta("repo_root", str(repository))
        reader = forbid_reads(monkeypatch)
        store.sync(graph)
        store.sync(graph)
        assert stored_hash(store, "pkg/service.py") == value
        assert graph == original
        reader.assert_not_called()


def test_repo_root_metadata_is_read_once_per_sync(repository, monkeypatch):
    with closing(Store(repository / ".blastradius/index.db")) as store:
        store.set_meta("repo_root", str(repository))
        get_meta = Mock(wraps=store.get_meta)
        monkeypatch.setattr(store, "get_meta", get_meta)
        store.sync(
            {"nodes": [{"id": "pkg/service.py"}, {"id": "missing.py"}], "links": []}
        )
        get_meta.assert_called_once_with("repo_root")
