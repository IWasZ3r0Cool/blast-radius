# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
# Modified by Christopher Wilson.
from __future__ import annotations

"""Git history backfill via plumbing — never touches the working tree.

Dependency rule: must not import from blastradius.graph or blastradius.semantic.
"""

import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryFile
from threading import Event, Timer
from typing import BinaryIO

from blastradius.import_graph import ImportGraph, Snapshot, report_diagnostics
from blastradius.import_graph.parsing import ParsedSource, parse
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


def _tree_blobs(root: Path, commit_hash: str) -> dict[str, tuple[str, str]]:
    """Read path -> (mode, blob hash) for regular files and symlinks."""
    result = subprocess.run(
        ["git", "ls-tree", "-rz", "--full-tree", commit_hash],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=60,
        check=True,
    )
    tree = {}
    for entry in result.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, blob_hash = meta.split()
        if kind == "blob" and mode in {"100644", "100755", "120000"}:
            tree[path] = (mode, blob_hash)
    return tree


def git_ls_tree(root: Path, commit_hash: str) -> dict[str, str]:
    """Return the regular-file blobs used by the historical import extractor."""
    return {
        path: blob
        for path, (mode, blob) in _tree_blobs(root, commit_hash).items()
        if mode != "120000"
    }


def git_tracked_paths(root: Path, commit_hash: str) -> set[str]:
    """Return tracked file and symlink paths, excluding submodule entries."""
    return set(_tree_blobs(root, commit_hash))


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
        # Clean up the child even on KeyboardInterrupt/SystemExit, then let the
        # original interruption propagate rather than leaving Git running.
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


def _invalid_path_part(part: str) -> bool:
    return Path(part).name != part or (
        os.name == "nt"
        and (
            ":" in part
            or part != part.rstrip(" .")
            or PureWindowsPath(part).is_reserved()
        )
    )


def _read_tree_blobs(
    stream: BinaryIO, tree: dict[str, tuple[str, str]], destination: Path
) -> dict[Path, str]:
    symlinks = {}
    for path, (mode, blob) in tree.items():
        parts = path.split("/")
        if any(part in {"", ".", ".."} or _invalid_path_part(part) for part in parts):
            raise ValueError(f"Unsupported Git path: {path!r}")

        header = stream.readline(256)
        fields = header.split()
        if (
            not header.endswith(b"\n")
            or len(fields) != 3
            or fields[:2] != [blob.encode("ascii"), b"blob"]
            or not fields[2].isdigit()
        ):
            raise RuntimeError(f"Invalid git cat-file response for {path!r}")
        size = int(fields[2])

        target = destination.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Placeholders let symlink validation use the temporary filesystem's own
        # case and filename matching before any symlink can be followed.
        with target.open("xb") as file:
            if mode == "120000":
                if size > 64 * 1024:
                    raise ValueError(f"Git symlink target is too long: {path!r}")
                data = stream.read(size)
                if len(data) != size:
                    raise RuntimeError(f"Incomplete Git blob for {path!r}")
                symlinks[target] = os.fsdecode(data)
            else:
                remaining = size
                while remaining:
                    data = stream.read(min(remaining, 1024 * 1024))
                    if not data:
                        raise RuntimeError(f"Incomplete Git blob for {path!r}")
                    file.write(data)
                    remaining -= len(data)
        if stream.read(1) != b"\n":
            raise RuntimeError(f"Missing Git blob terminator for {path!r}")
    return symlinks


def _file_identity(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino


def _safe_link_destination(
    root: Path, path: Path, targets: dict[tuple[int, int], str]
) -> Path | None:
    """Resolve a chain using the regular-file placeholders in the temporary tree."""
    parts = list(path.parent.relative_to(root).parts)
    remaining = [path.name]
    expansions = 0
    while remaining:
        part = remaining.pop()
        if part == "..":
            if not parts:
                return None
            parts.pop()
            continue
        if _invalid_path_part(part):
            return None
        try:
            target = targets.get(_file_identity(root.joinpath(*parts, part)))
        except (FileNotFoundError, NotADirectoryError):
            target = None
        if target is None:
            parts.append(part)
            continue

        expansions += 1
        link = Path(target)
        if expansions > 64 or not target or "\0" in target or link.anchor:
            return None
        remaining.extend(reversed(link.parts))
    return root.joinpath(*parts)


def materialize_commit(root: Path, commit_hash: str, destination: Path) -> None:
    """Stream tracked blobs into an empty temporary directory for full analysis."""
    tree = _tree_blobs(root, commit_hash)
    if not tree:
        return

    command = ["git", "cat-file", "--batch"]
    timeout = 60
    timed_out = Event()
    # Use files for requests and stderr so the child can stream stdout without
    # blocking on either of the other channels.
    with TemporaryFile() as requests, TemporaryFile() as errors:
        for _, blob in tree.values():
            requests.write(blob.encode("ascii") + b"\n")
        requests.seek(0)
        with subprocess.Popen(
            command, cwd=root, stdin=requests, stdout=subprocess.PIPE, stderr=errors
        ) as proc:
            assert proc.stdout is not None

            def expire() -> None:
                timed_out.set()
                proc.kill()

            timer = Timer(timeout, expire)
            timer.start()
            try:
                symlinks = _read_tree_blobs(proc.stdout, tree, destination)
                if proc.stdout.read(1):
                    raise RuntimeError("Unexpected trailing git cat-file output")
                if proc.wait():
                    errors.seek(0)
                    detail = errors.read(4096).decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"git cat-file failed: {detail}")
            except RuntimeError as exc:
                if timed_out.is_set():
                    raise subprocess.TimeoutExpired(command, timeout) from exc
                raise
            finally:
                timer.cancel()
                timer.join()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    targets = {_file_identity(path): target for path, target in symlinks.items()}
    safe_links = []
    for path, target in symlinks.items():
        resolved = _safe_link_destination(destination, path, targets)
        if resolved is not None:
            safe_links.append((path, target, os.name == "nt" and resolved.is_dir()))
    for path in symlinks:
        path.unlink()
    for path, target, is_dir in safe_links:
        path.symlink_to(target, target_is_directory=is_dir)


# ── backfill ──────────────────────────────────────────────────────────────────


def _extract_graph(
    root: Path,
    tree: dict[str, str],
    blob_cache: OrderedDict[str, bytes],
    parsed_cache: dict[str, tuple[str, ParsedSource]],
) -> tuple[set[str], set[tuple[str, str]], set[str]]:
    # Prefetch bytes in batches, including common configuration files.
    hashes = {
        blob
        for path, blob in tree.items()
        if not skipped(path)
        and (
            PurePosixPath(path).suffix.lower()
            in SOURCE_EXTENSIONS | {".json", ".jsonc"}
            or PurePosixPath(path).name in {"go.mod", "go.work", "Cargo.toml"}
        )
    }
    # skipped() excludes dotfiles; .gitignore is nevertheless configuration.
    if ".gitignore" in tree:
        hashes.add(tree[".gitignore"])
    contents = {blob: blob_cache[blob] for blob in hashes if blob in blob_cache}
    contents.update(_git_cat_file_batch(root, sorted(hashes - contents.keys())))

    def read_blob(path: str) -> bytes:
        blob = tree[path]
        if blob not in contents:
            # Configuration extends may refer to any file advertised by the snapshot.
            if blob in blob_cache:
                contents[blob] = blob_cache[blob]
            else:
                contents.update(_git_cat_file_batch(root, [blob]))
        return contents[blob]

    snapshot = Snapshot(set(tree), read_blob)
    # Retain syntax only for the current tree. Resolve it afresh so unchanged
    # importers can acquire new targets or follow changed configuration.
    parsed = {
        path: (
            parsed_cache[path]
            if path in parsed_cache and parsed_cache[path][0] == tree[path]
            else (tree[path], parse(path, snapshot.read(path)))
        )
        for path in sorted(snapshot.files)
        if PurePosixPath(path).suffix in SOURCE_EXTENSIONS
    }
    parsed_cache.clear()
    parsed_cache.update(parsed)
    graph = ImportGraph(
        snapshot,
        parsed_sources={path: entry[1] for path, entry in parsed.items()},
    )
    nodes, edges = graph.graph()
    # Include on-demand reads in the cross-commit cache as well.
    for blob, content in contents.items():
        blob_cache[blob] = content
        blob_cache.move_to_end(blob)
    while len(blob_cache) > 512:
        blob_cache.popitem(last=False)
    return nodes, edges, graph.diagnostics


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
    parsed_cache: dict[str, tuple[str, ParsedSource]] = {}
    diagnostics: set[str] = set()

    processed = 0

    for commit in commits:
        h = commit["hash"]
        curr_tree = git_ls_tree(root, h)

        # ── file temporal tracking ────────────────────────────────────────────
        for path in curr_tree:
            if path not in file_first_seen:
                file_first_seen[path] = h
            file_last_seen_alive[path] = h

        nodes, edges, commit_diagnostics = _extract_graph(
            root, curr_tree, blob_cache, parsed_cache
        )
        diagnostics.update(commit_diagnostics)
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
    files_tracked = len(file_first_seen)
    print(
        f"Backfill complete: {processed} commits, {files_tracked} files tracked.",
        file=sys.stderr,
    )
    return processed, files_tracked
