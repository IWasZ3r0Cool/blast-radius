# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from collections.abc import Iterable


def filter_changed_since(
    result: dict, modified: Iterable[str], *, touched_packages: Iterable[str] = ()
) -> None:
    """Add modified paths and keep edges touching changed files or their packages."""
    structural = set(result["added_files"]) | set(result["removed_files"])
    result["modified_files"] = [path for path in modified if path not in structural]
    touched = structural.union(result["modified_files"], touched_packages)
    suppressed = 0
    for key in ("added_edges", "removed_edges"):
        edges = result[key]
        filtered = [
            e for e in edges if e["source"] in touched or e["target"] in touched
        ]
        result[key] = filtered
        suppressed += len(edges) - len(filtered)
    if suppressed:
        result["suppressed_edge_count"] = suppressed
