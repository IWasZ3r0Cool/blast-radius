# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Git history backfill via plumbing — never touches the working tree.

Dependency rule: must not import from blastradius.graph or blastradius.semantic.
"""

import subprocess
import sys
from collections import OrderedDict
from pathlib import Path, PurePosixPath

from blastradius.import_graph import ImportGraph, Snapshot, report_diagnostics
from blastradius.import_graph.snapshot import SOURCE_EXTENSIONS, skipped

# ── git plumbing helpers ──────────────────────────────────────────────────────


def _run(cmd: list, cwd: Path, timeout: int = 60) -> str:
    r = subprocess.run(
        cmd,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        timeout=timeout,
        check=True,
    )
    return r.stdout


def git_log(
    root: Path,
    since: str | None = None,
    max_count: int | None = None,
) -> list[dict]:
    """Return commits oldest→newest within the requested bounds."""
    cmd = [
        "git",
        "log",
        "--format=%H%x00%ae%x00%aI%x00%cI%x00%P%x00%s",
        "--topo-order",
        "--reverse",
    ]
    if since:
        cmd.append(f"--since={since}")
    if max_count:
        cmd.extend(["-n", str(max_count)])

    out = _run(cmd, root, timeout=120)
    commits = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\x00")
        if len(parts) < 6:
            continue
        hash_, author, authored_at, committed_at, parents, message = parts[:6]
        parent_hash = parents.split()[0] if parents.strip() else None
        commits.append(
            {
                "hash": hash_,
                "author": author,
                "authored_at": authored_at,
                "committed_at": committed_at,
                "parent_hash": parent_hash,
                "message": message,
            }
        )
    return commits


def git_ls_tree(root: Path, commit_hash: str) -> dict[str, str]:
    """Return {repo_relative_path: blob_hash} for all files at commit."""
    out = _run(["git", "ls-tree", "-rz", "--full-tree", commit_hash], root)
    tree = {}
    for entry in out.split("\0"):
        if "\t" not in entry:
            continue
        meta, path = entry.split("\t", 1)
        parts = meta.split()
        if len(parts) == 3 and parts[0] in {"100644", "100755"} and parts[1] == "blob":
            tree[path] = parts[2]
    return tree


def _git_cat_file_batch(root: Path, blob_hashes: list) -> dict[str, bytes]:
    """Read multiple blobs via git cat-file --batch without touching working tree."""
    if not blob_hashes:
        return {}

    proc = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    inp = "\n".join(blob_hashes) + "\n"
    try:
        stdout, stderr = proc.communicate(inp.encode(), timeout=60)
    except BaseException:
        proc.kill()
        proc.communicate()
        raise
    if proc.returncode:
        raise RuntimeError(
            f"git cat-file failed: {stderr.decode('utf-8', errors='replace')}"
        )

    results: dict[str, bytes] = {}
    pos = 0
    data = stdout

    for blob_hash in blob_hashes:
        nl = data.find(b"\n", pos)
        if nl == -1:
            break
        header = data[pos:nl].decode("utf-8", errors="replace")
        pos = nl + 1
        parts = header.split()
        if len(parts) < 3 or parts[1] == "missing":
            continue
        try:
            size = int(parts[2])
        except ValueError:
            continue
        content_bytes = data[pos : pos + size]
        results[blob_hash] = content_bytes
        pos += size + 1  # trailing newline after blob content

    if len(results) != len(set(blob_hashes)):
        raise RuntimeError("git cat-file returned incomplete blob data")
    return results


# ── backfill ──────────────────────────────────────────────────────────────────


def backfill(
    root: Path,
    store: object,  # blastradius.store.Store — avoid circular import
    since: str | None = None,
    max_commits: int | None = 1000,
) -> tuple[int, int]:
    """Walk git history oldest→newest, populate temporal data without touching cwd.

    Returns (commits_processed, files_tracked).
    """
    commits = git_log(root, since=since, max_count=max_commits)
    if not commits:
        print("No commits found.", file=sys.stderr)
        return 0, 0

    print(f"Backfilling {len(commits)} commit(s)…", file=sys.stderr)

    # Record commits in DB first
    for c in commits:
        store.record_commit(c)

    # Forward pass: oldest→newest
    # Track per-path: first commit seen, last commit seen (while alive)
    file_first_seen: dict[str, str] = {}  # path → hash
    file_last_seen_alive: dict[str, str] = {}  # path → hash (last commit where present)

    # Track edge presence: (source, target, kind) → first hash, last alive hash
    edge_first_seen: dict[tuple, str] = {}
    edge_last_seen_alive: dict[tuple, str] = {}

    blob_cache: OrderedDict[str, bytes] = OrderedDict()
    diagnostics: set[str] = set()
    unsupported: set[str] = set()

    processed = 0

    for commit in commits:
        h = commit["hash"]
        curr_tree = git_ls_tree(root, h)
        known_files = set(curr_tree.keys())

        # ── file temporal tracking ────────────────────────────────────────────
        for path in curr_tree:
            if path not in file_first_seen:
                file_first_seen[path] = h
            file_last_seen_alive[path] = h

        # Parse cached source bytes, but resolve against the entire current
        # inventory/configuration. Unchanged importers can acquire new targets.
        readable = [
            path
            for path in curr_tree
            if not skipped(path)
            and (
                PurePosixPath(path).suffix.lower()
                in SOURCE_EXTENSIONS | {".json", ".jsonc"}
                or path == ".gitignore"
            )
        ]
        # skipped() excludes dotfiles; .gitignore is nevertheless configuration.
        if ".gitignore" in curr_tree and ".gitignore" not in readable:
            readable.append(".gitignore")
        hashes = {curr_tree[path] for path in readable}
        contents = {blob: blob_cache[blob] for blob in hashes if blob in blob_cache}
        contents.update(_git_cat_file_batch(root, sorted(hashes - contents.keys())))
        for blob, content in contents.items():
            blob_cache[blob] = content
            blob_cache.move_to_end(blob)
        while len(blob_cache) > 512:
            blob_cache.popitem(last=False)

        snapshot = Snapshot(
            known_files,
            lambda path, contents=contents, tree=curr_tree: contents[tree[path]],
        )
        graph = ImportGraph(snapshot)
        nodes, edges = graph.graph()
        diagnostics.update(graph.diagnostics)
        unsupported.update(
            PurePosixPath(path).suffix
            for path in snapshot.files
            if PurePosixPath(path).suffix in {".go", ".rs"}
        )
        store.record_graph_snapshot(h, nodes, edges)
        for source, target in edges:
            key = (source, target, "imports")
            edge_first_seen.setdefault(key, h)
            edge_last_seen_alive[key] = h

        processed += 1

        if processed % 100 == 0:
            print(f"  {processed}/{len(commits)} commits processed", file=sys.stderr)

    # ── write temporal data to DB ─────────────────────────────────────────────
    store.apply_file_temporal(file_first_seen, file_last_seen_alive)
    store.apply_edge_temporal(edge_first_seen, edge_last_seen_alive)

    report_diagnostics(diagnostics)
    if unsupported:
        print(
            "History import coverage: Python and JS/TS/Vue; Go/Rust dependencies are not backfilled.",
            file=sys.stderr,
        )
    files_tracked = len(file_first_seen)
    print(
        f"Backfill complete: {processed} commits, {files_tracked} files tracked.",
        file=sys.stderr,
    )
    return processed, files_tracked
