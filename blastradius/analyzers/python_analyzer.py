# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
# Modified by Christopher Wilson.
from __future__ import annotations

"""Python repository analyzer (AST-based)."""

from pathlib import Path

from blastradius.import_graph import ImportGraph, Snapshot, report_diagnostics

from .base import dir_group

CONFIG_NAMES = {"config", "settings", "constants", "env", "configuration", "conf"}


def node_type(path: Path) -> str:
    return "config" if path.stem.lower() in CONFIG_NAMES else "module"


def analyze(root: Path, group_map: dict):
    """
    Returns (nodes, external_nodes, links_map, meta).
    links_map keys are (source_rel, target_rel) tuples.
    """
    snapshot = Snapshot.directory(root)
    py_files = [
        root / path for path in sorted(snapshot.files) if Path(path).suffix == ".py"
    ]
    imports = ImportGraph(snapshot)
    nodes = []
    links_map: dict[tuple[str, str], int] = {}
    external_nodes: dict[str, dict] = {}
    total_loc = 0

    for f in py_files:
        rel = f.relative_to(root).as_posix()
        try:
            # Let Python's parser honor UTF-8, BOMs, and source encoding cookies.
            source = f.read_bytes()
        except OSError:
            continue

        loc = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n").count(b"\n") + 1
        total_loc += loc

        result = imports.analyze(rel, source)

        nodes.append(
            {
                "id": rel,
                "type": node_type(f),
                "language": "python",
                "size": loc,
                "loc": loc,
                "group": dir_group(f, root, group_map),
                "imports": result.count,
            }
        )

        for target in result.targets:
            key = (rel, target)
            links_map[key] = links_map.get(key, 0) + 1
        for mod in result.external:
            top_level = mod.split(".")[0]
            if top_level not in external_nodes:
                external_nodes[top_level] = {
                    "id": top_level,
                    "type": "import",
                    "language": "python",
                    "size": 40,
                    "loc": 0,
                    "group": 9000,
                    "imports": 0,
                }
            key = (rel, top_level)
            links_map[key] = links_map.get(key, 0) + 1

    report_diagnostics(imports.diagnostics)

    return (
        nodes,
        list(external_nodes.values()),
        links_map,
        {
            "total_files": len(py_files),
            "total_loc": total_loc,
        },
    )
