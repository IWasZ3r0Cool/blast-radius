# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Go repository analyzer with package-level dependency nodes."""

from pathlib import Path, PurePosixPath

from blastradius.import_graph import Snapshot, report_diagnostics
from blastradius.import_graph.go import GoGraph, _scan_go_source
from blastradius.import_graph.go import external_pkg_name as _external_pkg_name

from .base import is_ignored, is_skip_dir

_ROUTE_DIRS = {
    "handlers",
    "controllers",
    "routes",
    "api",
    "endpoints",
    "http",
    "server",
}
_STORE_DIRS = {
    "models",
    "store",
    "storage",
    "repository",
    "repos",
    "db",
    "database",
    "data",
    "dao",
}
_CONFIG_DIRS = {"config", "cfg", "configuration", "settings", "conf"}


def collect_files(root: Path, patterns: list):
    files = []
    for path in root.rglob("*.go"):
        if is_skip_dir(path.relative_to(root)) or is_ignored(path, root, patterns):
            continue
        files.append(path)
    return sorted(files)


def external_pkg_name(imp_path: str) -> str:
    return _external_pkg_name(imp_path)


def extract_imports(source: str):
    return _scan_go_source(source)[1]


def parse_module_name(root: Path):
    module = GoGraph(Snapshot.directory(root)).modules.get(PurePosixPath("."))
    return module.name if module else None


def pkg_node_type(pkg_path: str, pkg_name: str) -> str:
    parts = pkg_path.lower().split("/")
    name = pkg_name.lower()
    if any(part in _CONFIG_DIRS for part in parts) or name in _CONFIG_DIRS:
        return "config"
    if any(part in _ROUTE_DIRS for part in parts) or name in _ROUTE_DIRS:
        return "route"
    if any(part in _STORE_DIRS for part in parts) or name in _STORE_DIRS:
        return "store"
    return "module"


def analyze(root: Path, group_map: dict):
    graph = GoGraph(Snapshot.directory(root))
    nodes, external_nodes, links = [], {}, {}
    for node_id, result in graph.results.items():
        directory = "" if node_id == "." else node_id
        top_key = directory.split("/")[0]
        if top_key not in group_map:
            group_map[top_key] = len(group_map)
        nodes.append(
            {
                "id": node_id,
                "type": pkg_node_type(directory, result.name),
                "language": "go",
                "size": result.loc,
                "loc": result.loc,
                "group": group_map[top_key],
                "imports": result.imports.count,
            }
        )
        for target in result.imports.targets + result.imports.external:
            key = (node_id, target)
            links[key] = links.get(key, 0) + 1
        for target in result.imports.external:
            external_nodes.setdefault(
                target,
                {
                    "id": target,
                    "type": "import",
                    "language": "go",
                    "size": 40,
                    "loc": 0,
                    "group": 9000,
                    "imports": 0,
                },
            )
    report_diagnostics(graph.diagnostics)
    return (
        nodes,
        list(external_nodes.values()),
        links,
        {
            "total_files": sum(len(r.sources) for r in graph.results.values()),
            "total_loc": sum(r.loc for r in graph.results.values()),
        },
    )
