# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Build a Markdown snapshot of an indexed dependency graph."""

import json
import re
import subprocess
from contextlib import closing
from pathlib import Path

from blastradius.impact import is_file_node, rank_blast_files
from blastradius.index import (
    CLEAN_INDEX_META_KEY,
    clean_index_stamp,
    db_path_for,
    find_index,
    git_modified,
    git_resolve,
)
from blastradius.store import Store
from blastradius.temporal.changes import filter_changed_since


def _code(value: str) -> str:
    # Keep unusual Git filenames on one Markdown line.
    value = re.sub(
        r"[\x00-\x1f\x7f\ud800-\udfff]",
        lambda match: match[0].encode("unicode_escape").decode("ascii"),
        value,
    )
    fence = "`" * (1 + max(map(len, re.findall(r"`+", value)), default=0))
    if value.startswith(("`", " ")) or value.endswith(("`", " ")):
        value = f" {value} "
    return f"{fence}{value}{fence}"


def _cycles(nodes: list[dict], links: list[dict]) -> list[list[str]]:
    """Find cyclic strongly connected components using two iterative DFS passes."""
    outgoing = {node["id"]: set() for node in nodes if is_file_node(node)}
    incoming = {path: set() for path in outgoing}
    for link in links:
        source, target = link["source"], link["target"]
        if source in outgoing and target in outgoing:
            outgoing[source].add(target)
            incoming[target].add(source)

    visited: set[str] = set()
    finished: list[str] = []
    for path, targets in outgoing.items():
        if path in visited:
            continue
        visited.add(path)
        stack = [(path, iter(targets))]
        while stack:
            node, neighbors = stack[-1]
            neighbor = next(neighbors, None)
            if neighbor is None:
                finished.append(node)
                stack.pop()
            elif neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, iter(outgoing[neighbor])))

    visited.clear()
    cycles = []
    for path in reversed(finished):
        if path in visited:
            continue
        group = []
        pending = [path]
        visited.add(path)
        while pending:
            node = pending.pop()
            group.append(node)
            for neighbor in incoming[node] - visited:
                visited.add(neighbor)
                pending.append(neighbor)
        if len(group) > 1 or path in outgoing[path]:
            cycles.append(sorted(group))
    return sorted(cycles)


def _compare_commits(
    store: Store, root: Path, base: str, head: str
) -> tuple[str, dict]:
    from blastradius.import_graph import EXTRACTION_VERSION, HISTORY_COVERAGE
    from blastradius.temporal.history import git_tracked_paths, graph_at_commit

    before = store.snapshot_graph(base)
    after = store.snapshot_graph(head)
    if (
        before is not None
        and after is not None
        and before.coverage == after.coverage
        and before.extraction_version == after.extraction_version == EXTRACTION_VERSION
    ):
        # Full index snapshots may also contain virtual services, jobs, and tables.
        base_paths = before.paths & git_tracked_paths(root, base)
        head_paths = after.paths & git_tracked_paths(root, head)
        base_edges = {
            (s, t) for s, t in before.edges if s in base_paths and t in base_paths
        }
        head_edges = {
            (s, t) for s, t in after.edges if s in head_paths and t in head_paths
        }
        coverage = before.coverage
    else:
        # Extract both sides the same way to avoid reporting extractor differences.
        base_paths, base_edges = graph_at_commit(root, base)
        head_paths, head_edges = graph_at_commit(root, head)
        coverage = HISTORY_COVERAGE

    changes = {
        "added_files": sorted(head_paths - base_paths),
        "removed_files": sorted(base_paths - head_paths),
        "added_edges": [
            {"source": s, "target": t} for s, t in sorted(head_edges - base_edges)
        ],
        "removed_edges": [
            {"source": s, "target": t} for s, t in sorted(base_edges - head_edges)
        ],
    }
    filter_changed_since(changes, sorted(git_modified(root, base, head, strict=True)))
    return coverage, changes


def _change_sections(root: Path, graph_text: str, ref: str) -> list[str]:
    base = git_resolve(root, ref)
    if base is None:
        raise ValueError(f"Could not resolve commit ref: {ref!r}")
    head = git_resolve(root, "HEAD")
    if head is None:
        raise ValueError("Could not resolve HEAD to a commit")
    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        cwd=root,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout
    if prefix.strip():
        raise ValueError(
            "For --ref, run blastradius analyze . and blastradius review "
            "from the Git repository root."
        )

    db_path = db_path_for(root)
    if not db_path.is_file():
        raise FileNotFoundError(
            "No .blastradius/index.db found. Run: blastradius analyze ."
        )
    with closing(Store(db_path)) as store:
        if store.get_meta(CLEAN_INDEX_META_KEY) != clean_index_stamp(head, graph_text):
            raise ValueError(
                "The graph is not an index of clean HEAD. "
                "From the repository root, run blastradius analyze . on a clean checkout. "
                "Remove or isolate any ignored files that affect the dependency graph."
            )
        coverage, changes = _compare_commits(store, root, base, head)

    coverage_text = (
        "stored index snapshots (tracked file nodes)"
        if coverage == "index"
        else f"historical import graphs ({', '.join(coverage.split(','))})"
    )
    lines = [
        "## Committed changes",
        "",
        f"- Base: {_code(base)}",
        f"- HEAD: {_code(head)}",
        f"- Graph coverage: {coverage_text}",
    ]
    for key, title in (
        ("added_files", "Added files"),
        ("removed_files", "Removed files"),
        ("modified_files", "Modified files"),
        ("added_edges", "Added edges"),
        ("removed_edges", "Removed edges"),
    ):
        items = changes[key]
        lines.extend(["", f"### {title} ({len(items)})", ""])
        if key.endswith("_edges"):
            lines.extend(
                f"- {_code(e['source'])} → {_code(e['target'])}" for e in items
            )
        else:
            lines.extend(f"- {_code(path)}" for path in items)
        if not items:
            lines.append("None.")
    if suppressed := changes.get("suppressed_edge_count"):
        lines.extend(
            ["", f"Edge changes outside the changed file set omitted: {suppressed}."]
        )
    return lines


def build_review(start: Path, *, ref: str | None = None) -> str:
    """Discover an index and render the graph and optional committed changes."""
    index = find_index(start)
    if index is None or not index.is_file():
        raise FileNotFoundError("No blastradius.json found. Run: blastradius analyze .")
    graph_text = index.read_text(encoding="utf-8")
    data = json.loads(graph_text)
    if not isinstance(data, dict) or not all(
        isinstance(data.get(key), list) for key in ("nodes", "links")
    ):
        raise ValueError("Invalid blastradius.json graph. Run: blastradius analyze .")

    nodes = sorted(data["nodes"], key=lambda node: node["id"])
    ranked = rank_blast_files(nodes, threshold=0)[:20]
    lines = ["# Blast Radius review", "", "## Top blast scores", ""]
    lines.extend(
        f"{i}. {_code(node['id'])} — {node.get('blast_score', 0):.1f} "
        f"({node.get('direct_dependents', 0)} direct, "
        f"{node.get('transitive_dependents', 0)} transitive)"
        for i, node in enumerate(ranked, start=1)
    )
    if not ranked:
        lines.append("No indexed files.")

    cycles = _cycles(nodes, data["links"])
    lines.extend(["", f"## Dependency cycle groups ({len(cycles)})", ""])
    lines.extend("- " + ", ".join(map(_code, group)) for group in cycles)
    if not cycles:
        lines.append("None.")

    if ref is not None:
        lines.extend(["", *_change_sections(index.parent, graph_text, ref)])
    return "\n".join(lines) + "\n"
