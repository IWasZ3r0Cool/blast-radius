# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
# Modified by Christopher Wilson.
from __future__ import annotations

"""Rust repository analyzer."""

from pathlib import Path, PurePosixPath

from blastradius.import_graph import Snapshot, report_diagnostics
from blastradius.import_graph.rust import (
    STDLIB_CRATES as _STDLIB_CRATES,
)
from blastradius.import_graph.rust import (
    RustGraph,
    _ModDecl,
    _module_candidate,
    read_manifest,
    resolve_crate_use,
)

from .base import dir_group, is_ignored, is_skip_dir

STDLIB_CRATES = _STDLIB_CRATES

_ROUTE_DIRS = {"routes", "handlers", "controllers", "endpoints", "api"}
_STORE_DIRS = {"models", "db", "database", "storage", "repository", "repos", "store"}
_CONFIG_NAMES = {"config", "settings", "configuration", "constants", "env"}


def collect_files(root: Path, patterns: list):
    files = []
    for path in root.rglob("*.rs"):
        if is_skip_dir(path.relative_to(root)) or is_ignored(path, root, patterns):
            continue
        files.append(path)
    return sorted(files)


def parse_cargo_deps(root: Path):
    data = read_manifest(Snapshot.directory(root), "Cargo.toml", set())
    return {
        alias
        for section in ("dependencies", "dev-dependencies", "build-dependencies")
        for alias in data.get(section, {})
    }


def node_type(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    if stem in _CONFIG_NAMES or any(part in _CONFIG_NAMES for part in parts):
        return "config"
    if any(part in _ROUTE_DIRS for part in parts):
        return "route"
    if any(part in _STORE_DIRS for part in parts):
        return "store"
    return "module"


def resolve_mod_decl(mod_name: str, file_path: Path, root: Path, all_files: set):
    relative = PurePosixPath(file_path.relative_to(root).as_posix())
    return _module_candidate(
        relative,
        _ModDecl(mod_name, ()),
        relative.name in {"lib.rs", "main.rs"},
        all_files,
        PurePosixPath("."),
    )


def resolve_use_internal(use_path: str, root: Path, all_files: set):
    return resolve_crate_use(use_path, all_files)


def analyze(root: Path, group_map: dict):
    graph = RustGraph(Snapshot.directory(root))
    nodes, external_nodes, links = [], {}, {}
    for filename, result in graph.results.items():
        path = root / filename
        nodes.append(
            {
                "id": filename,
                "type": node_type(path.relative_to(root)),
                "language": "rust",
                "size": result.loc,
                "loc": result.loc,
                "group": dir_group(path, root, group_map),
                "imports": result.imports.count,
            }
        )
        for target in result.imports.targets + result.imports.external:
            key = (filename, target)
            links[key] = links.get(key, 0) + 1
        for target in result.imports.external:
            external_nodes.setdefault(
                target,
                {
                    "id": target,
                    "type": "import",
                    "language": "rust",
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
            "total_files": len(graph.results),
            "total_loc": sum(r.loc for r in graph.results.values()),
        },
    )
