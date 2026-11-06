# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Conservative source references plus symbol-filtered file graph traversal."""

import ast
from collections import deque
from pathlib import Path


def _module(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def python_usage(source: str, importer: str, target: str, name: str) -> dict:
    tree = ast.parse(source)
    bindings = set()
    imported = False
    uncertain = False
    target_module = _module(target)

    def matches_module(module):
        return bool(module) and (
            target_module == module or target_module.endswith("." + module)
        )

    package = _module(importer).split(".")
    if Path(importer).name != "__init__.py":
        package = package[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = ".".join(
                    package[: len(package) - node.level + 1]
                    + ([module] if module else [])
                )
            for alias in node.names:
                if matches_module(module + "." + alias.name):
                    imported = True
                    bindings.add((alias.asname or alias.name, name))
            if not matches_module(module):
                continue
            for alias in node.names:
                if alias.name == name:
                    imported = True
                    bindings.add((alias.asname or name,))
                elif alias.name == "*":
                    imported = True
                    uncertain = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if matches_module(alias.name):
                    imported = True
                    bindings.add(
                        tuple((alias.asname or alias.name).split(".")) + (name,)
                    )

    # Conditional/type-only imports cannot establish runtime availability.
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.If, ast.Try, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and any(
            isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node)
        ):
            uncertain = True

    references = []

    def chain(node):
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            return chain(node.value) + (node.attr,)
        return ()

    def visit(node, scope=None):
        nonlocal uncertain
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
            if node.name in {binding[0] for binding in bindings}:
                uncertain = True
        # Shadowed bindings require name resolution beyond this conservative pass.
        if isinstance(node, (ast.Name, ast.arg)):
            identifier = node.id if isinstance(node, ast.Name) else node.arg
            if identifier in {binding[0] for binding in bindings} and (
                isinstance(node, ast.arg) or isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                uncertain = True
        if (
            isinstance(node, (ast.Name, ast.Attribute))
            and isinstance(node.ctx, ast.Load)
            and chain(node) in bindings
        ):
            references.append(
                {
                    "line": node.lineno,
                    "column": node.col_offset,
                    "symbol": scope or "<module>",
                }
            )
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree)
    return {
        "imports_symbol": imported,
        "usage": "unknown" if uncertain else ("referenced" if references else "unused"),
        "references": sorted(references, key=lambda r: (r["line"], r["column"])),
    }


def symbol_impact(store, root: Path, file: str, name: str) -> dict:
    importers = []
    for importer in sorted(set(store.importers_of_file(file))):
        result = {
            "file": importer,
            "imports_symbol": None,
            "usage": "unknown",
            "references": [],
        }
        try:
            source_path = (root / importer).resolve()
            source_path.relative_to(root.resolve())
            if source_path.suffix == ".py" and Path(file).suffix == ".py":
                result.update(
                    python_usage(
                        source_path.read_text(encoding="utf-8"), importer, file, name
                    )
                )
        except (OSError, UnicodeError, SyntaxError, ValueError):
            pass
        importers.append(result)

    direct = {r["file"] for r in importers if r["usage"] == "referenced"}
    visited = {file} | direct
    queue = deque(sorted(direct))
    while queue:
        current = queue.popleft()
        for parent in store.importers_of_file(current):
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)
    transitive = visited - direct - {file}
    return {
        "importers": importers,
        "direct_ids": sorted(direct),
        "transitive_ids": sorted(transitive),
        "blast_score": len(direct) + 0.5 * len(transitive),
        "complete": all(r["usage"] != "unknown" for r in importers),
        "analysis": "Static Python import/reference analysis, not proof of runtime execution. "
        "Unknown usage is excluded from the score. Transitive edges are file-level; "
        "dynamic imports, re-exports and reachability are not resolved.",
    }
