# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Build and persist blastradius.json in the target repo root.

Phase-1 change: build() now also syncs graph data to a SQLite store at
<repo>/.blastradius/index.db.  The JSON write path is unchanged so existing
consumers keep working without modification.
"""

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from blastradius.analyze import analyze
from blastradius.impact import compute_blast_radius, enrich_links, enrich_nodes
from blastradius.store import Store
from blastradius.temporal.history import materialize_commit

INDEX_FILENAME = "blastradius.json"
CLEAN_INDEX_META_KEY = "clean_index"
_DB_DIR = ".blastradius"
_DB_NAME = "index.db"


def db_path_for(repo_root: Path) -> Path:
    return repo_root / _DB_DIR / _DB_NAME


def clean_index_stamp(commit: str, graph_text: str) -> str:
    """Tie the exported graph to the commit whose clean tree was indexed."""
    digest = hashlib.sha256(graph_text.encode("utf-8")).hexdigest()
    return f"v3:{commit}:{digest}"


def _git_head(root: Path) -> str | None:
    # Noninteractive Git children must never inherit MCP's live protocol pipe.
    # On Windows even inspecting that pipe during child startup can block.
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _worktree_is_clean(root: Path) -> bool:
    """Check for tracked changes and non-ignored untracked files."""
    try:
        outputs = {INDEX_FILENAME, "symbolindex.json"}
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            return False
        entries = iter(result.stdout.decode("utf-8", "surrogateescape").split("\0"))
        for entry in entries:
            if not entry:
                continue
            paths = [entry[3:]]
            if "R" in entry[:2] or "C" in entry[:2]:
                paths.append(next(entries, ""))
            for path in paths:
                # Ignore only known outputs. Any other change could affect an
                # analyzer, including Go/Rust or infrastructure configuration.
                if path in outputs or path.startswith(".blastradius/"):
                    continue
                return False
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _review_graph(
    data: dict,
) -> tuple[Counter[tuple[str, str | None]], set[tuple[str, str]]]:
    """Capture the graph facts used for rankings, cycles, and commit snapshots."""
    return (
        Counter((node["id"], node.get("type")) for node in data["nodes"]),
        {(link["source"], link["target"]) for link in data["links"]},
    )


def analyze_commit(root: Path, commit: str) -> dict:
    """Run the full analyzer on a commit without touching the worktree."""
    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=10,
        check=True,
    ).stdout.removesuffix(b"\n")
    with TemporaryDirectory(prefix="blastradius-commit-") as tmp:
        committed_tree = Path(tmp).resolve()
        materialize_commit(root, commit, committed_tree)
        return analyze(str(committed_tree / os.fsdecode(prefix)))


def _matches_committed_graph(root: Path, commit: str, data: dict) -> bool:
    # Git status omits ignored files. Compare the graph against the committed inputs.
    try:
        return _review_graph(data) == _review_graph(analyze_commit(root, commit))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not verify committed graph: {exc}", file=sys.stderr)
        return False


def _git_changed(root: Path, from_commit: str, to_commit: str) -> set:
    """Return repo-relative paths changed between two commits."""
    try:
        r = subprocess.run(
            ["git", "diff", "--name-status", from_commit, to_commit],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        paths = set()
        for line in r.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                paths.add(parts[-1])  # rename lines have 3 parts; last is dest
        return paths
    except Exception:  # noqa: BLE001
        return set()


def git_modified(
    root: Path,
    from_ref: str,
    to_ref: str = "HEAD",
    *,
    strict: bool = False,
    include_type_changes: bool = False,
) -> list[str]:
    """Return repo-relative modified paths, optionally including Git type changes."""
    try:
        r = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "-z",
                "--diff-filter=MT" if include_type_changes else "--diff-filter=M",
                from_ref,
                to_ref,
                "--",
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=True,
        )
        return [
            path
            for path in r.stdout.decode("utf-8", "surrogateescape").split("\0")
            if path
        ]
    except (OSError, subprocess.SubprocessError):
        if strict:
            raise
        return []


def git_reachable(root: Path, ref: str) -> set:
    """Return the set of commit hashes reachable from ref (for as-of queries)."""
    try:
        r = subprocess.run(
            ["git", "log", "--format=%H", ref],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if r.returncode != 0:
            return set()
        return {line.strip() for line in r.stdout.splitlines() if line.strip()}
    except Exception:  # noqa: BLE001
        return set()


def git_resolve(root: Path, ref: str) -> str | None:
    """Resolve a ref (branch, tag, partial hash) to a full commit hash."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class _EmbeddingConfig:
    endpoint: str
    model: str
    dims: int


def _embedding_config() -> _EmbeddingConfig | None:
    """Read embedding settings, returning None for missing or invalid values."""
    import os

    endpoint = os.environ.get("BLASTRADIUS_EMBEDDING_ENDPOINT", "")
    model = os.environ.get("BLASTRADIUS_EMBEDDING_MODEL", "")
    dims_str = os.environ.get("BLASTRADIUS_EMBEDDING_DIMS", "")
    if not (endpoint and model and dims_str):
        return None
    try:
        dims = int(dims_str)
    except ValueError:
        return None
    return _EmbeddingConfig(endpoint=endpoint, model=model, dims=dims)


def _embed_new_symbols(store) -> None:
    """Embed missing vectors; skip invalid configuration and warn on embedding failures."""
    config = _embedding_config()
    if config is None:
        return

    if not store.init_vectors(config.dims):
        return

    pairs = store.symbols_needing_embeddings()
    if not pairs:
        return

    try:
        from blastradius.semantic.provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            endpoint=config.endpoint, model=config.model, dims=config.dims
        )
        ids = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]
        vecs = provider.embed(texts)
        store.upsert_embeddings(list(zip(ids, vecs)))
        store.set_meta("embedding_model", config.model)
        store._conn.commit()
        print(f"Embedded {len(ids)} symbol(s)", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: embedding failed: {exc}", file=sys.stderr)


def build(repo_path: str, output: Path | None = None) -> dict:
    root = Path(repo_path).resolve()
    head_commit = _git_head(root)
    started_clean = head_commit is not None and _worktree_is_clean(root)
    data = analyze(str(root))

    blast = compute_blast_radius(data["nodes"], data["links"])
    enrich_nodes(data["nodes"], blast)
    enrich_links(data["nodes"], data["links"])

    data["meta"]["indexed"] = True
    graph_text = json.dumps(data, indent=2)

    # Detect changed file set for informational logging
    db_path = db_path_for(root)
    snapshot_matches_head = (
        head_commit is not None
        and started_clean
        and head_commit == _git_head(root)
        and _matches_committed_graph(root, head_commit, data)
        and _worktree_is_clean(root)
        and head_commit == _git_head(root)
    )
    changed_paths: set | None = None

    with closing(Store(db_path)) as store:
        store.set_meta("repo_root", str(root))

        last_commit = store.get_meta("last_indexed_commit")
        if last_commit and head_commit and last_commit != head_commit:
            changed_paths = _git_changed(root, last_commit, head_commit)
            print(
                f"Incremental: {len(changed_paths)} file(s) changed since {last_commit[:8]}",
                file=sys.stderr,
            )
        elif last_commit is None:
            print("Incremental: first index — full scan", file=sys.stderr)

        store.sync(
            data,
            commit=head_commit,
            changed_paths=changed_paths,
            snapshot=snapshot_matches_head,
        )

        if head_commit:
            store.set_meta("last_indexed_commit", head_commit)
        store.set_meta(
            CLEAN_INDEX_META_KEY,
            clean_index_stamp(head_commit, graph_text)
            if head_commit and snapshot_matches_head
            else "",
        )
        store._conn.commit()

        # Build and sync symbol index
        try:
            from blastradius.symbols import build_symbol_index

            symbol_data = build_symbol_index(str(root))
            store.sync_symbols(symbol_data, commit=head_commit)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: symbol sync failed: {exc}", file=sys.stderr)

        # Embed new symbols if an embedding endpoint is configured
        _embed_new_symbols(store)

    dest = output or (root / INDEX_FILENAME)
    dest.write_text(graph_text)

    meta = data["meta"]
    langs_str = ", ".join(meta.get("languages", ["unknown"]))
    print(
        f"Indexed {meta['total_files']} files, {meta['total_loc']} LOC "
        f"[{langs_str}] → {dest}",
        file=sys.stderr,
    )
    return data


def load(index_path: Path) -> dict:
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found — run: blastradius analyze <repo>"
        )
    return json.loads(index_path.read_text())


def find_index(start: Path) -> Path | None:
    """Walk up from start looking for blastradius.json."""
    current = start.resolve()
    for _ in range(10):
        candidate = current / INDEX_FILENAME
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def find_db(start: Path) -> Path | None:
    """Walk up from start looking for .blastradius/index.db."""
    current = start.resolve()
    for _ in range(10):
        candidate = current / _DB_DIR / _DB_NAME
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None
