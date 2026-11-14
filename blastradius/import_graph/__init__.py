# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
# Modified by Christopher Wilson.
from __future__ import annotations

"""Shared multi-language dependency analysis over working files or historical blobs."""

import posixpath
import sys
from collections.abc import Mapping
from pathlib import PurePosixPath

import json5

from .go import GoGraph
from .parsing import ParsedSource, Reference, parse
from .results import ImportResult as ImportResult
from .rust import RustGraph
from .snapshot import (
    JS_EXTENSIONS,
    SOURCE_EXTENSIONS,
    STYLE_EXTENSIONS,
    Snapshot,
    normalize,
)

EXTRACTION_VERSION = "4"
HISTORY_COVERAGE = "python,javascript,typescript,vue,go,rust"
_EXTENSIONS = (
    ".ts",
    ".tsx",
    ".d.ts",
    ".js",
    ".jsx",
    ".mts",
    ".cts",
    ".mjs",
    ".cjs",
    ".vue",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".styl",
)


def report_diagnostics(diagnostics) -> None:
    """Keep graph/JSON output clean and bound diagnostics on large histories."""
    messages = sorted(set(diagnostics))
    for message in messages[:10]:
        print(f"Warning: {message}", file=sys.stderr)
    if len(messages) > 10:
        print(f"Warning: {len(messages) - 10} more import diagnostics", file=sys.stderr)


class ImportGraph:
    """Resolve imports solely against a snapshot, with no installed-package lookup."""

    def __init__(
        self,
        snapshot: Snapshot,
        *,
        parsed_sources: Mapping[str, ParsedSource] | None = None,
    ):
        # Callers may reuse syntax from this snapshot; resolution stays local.
        self._parsed_sources = parsed_sources if parsed_sources is not None else {}
        self.snapshot = snapshot
        self.files = snapshot.files
        self.diagnostics: set[str] = set()
        self._json_cache: dict[str, dict] = {}
        self._configs: dict[str, tuple[str, dict]] = {}
        self._python_stems: set[str] = set()
        self.packages: dict[str, list[str]] = {}
        for path in sorted(self.files):
            if path.endswith(".py"):
                self._python_stems.add(path[:-3])
                directory = posixpath.dirname(path)
                while directory:
                    self._python_stems.add(directory)
                    directory = posixpath.dirname(directory)
            if PurePosixPath(path).name == "package.json":
                name = self._json(path).get("name")
                if isinstance(name, str):
                    self.packages.setdefault(name, []).append(posixpath.dirname(path))

    def _json(self, path: str) -> dict:
        if path not in self._json_cache:
            try:
                value = (
                    json5.loads(self.snapshot.read(path).decode("utf-8-sig"))
                    if path in self.snapshot.paths
                    else {}
                )
                if not isinstance(value, dict):
                    raise TypeError("expected an object")
            except (TypeError, ValueError, UnicodeError, OSError) as exc:
                self.diagnostics.add(f"{path}: invalid configuration: {exc}")
                value = {}
            self._json_cache[path] = value
        return self._json_cache[path]

    def _python(self, path: str, reference: Reference) -> set[str]:
        directory = posixpath.dirname(path)
        if reference.level:
            parts = directory.split("/") if directory else []
            source_root = int(
                parts[:1] == ["src"] and "src/__init__.py" not in self.files
            )
            if reference.level > len(parts) - source_root:
                return set()
            base = "/".join(parts[: len(parts) - reference.level + 1])
            stems = [posixpath.join(base, reference.specifier.replace(".", "/"))]
        else:
            module = reference.specifier.replace(".", "/")
            stems = [module, "src/" + module]
            if directory:
                stems.append(directory + "/" + module)

        def find(stem):
            stem = stem.rstrip("/")
            for candidate in (stem + ".py", stem + "/__init__.py"):
                if candidate in self.files:
                    return candidate
            return None

        for stem in stems:
            stem = stem.rstrip("/")
            result = set()
            module_file = find(stem)
            if module_file:
                result.add(module_file)
            if reference.kind == "from" and (
                module_file is None or module_file.endswith("/__init__.py")
            ):
                # A named import can name a submodule or an attribute. Only add
                # children backed by actual files; retain the package dependency.
                for name in reference.names:
                    if name != "*":
                        child = find(posixpath.join(stem, name))
                        if child:
                            result.add(child)
            if result:
                # Importing a regular package also executes its initializers.
                for target in tuple(result):
                    parent = posixpath.dirname(target)
                    while parent:
                        initializer = parent + "/__init__.py"
                        if initializer in self.files:
                            result.add(initializer)
                        parent = posixpath.dirname(parent)
                result.discard(path)
                return result
        return set()

    def _config(self, path: str) -> tuple[str, dict]:
        directory = posixpath.dirname(path)
        if directory in self._configs:
            return self._configs[directory]
        current = directory
        while True:
            for name in ("tsconfig.json", "jsconfig.json"):
                candidate = posixpath.join(current, name)
                if candidate in self.files:
                    result = self._load_config(candidate, set())
                    self._configs[directory] = result
                    return result
            if not current:
                self._configs[directory] = ("", {})
                return "", {}
            current = posixpath.dirname(current)

    def _load_config(self, path: str, seen: set[str]) -> tuple[str, dict]:
        if path in seen:
            self.diagnostics.add(f"{path}: cyclic configuration extends")
            return "", {}
        seen = seen | {path}
        directory = posixpath.dirname(path)
        data = self._json(path)
        base, inherited = "", {}
        extends = data.get("extends")
        if isinstance(extends, str):
            if extends.startswith("."):
                parent = normalize(posixpath.join(directory, extends))
                if (
                    parent
                    and parent not in self.files
                    and parent + ".json" in self.files
                ):
                    parent += ".json"
                if parent and parent in self.files:
                    base, inherited = self._load_config(parent, seen)
                else:
                    self.diagnostics.add(f"{path}: unresolved extends {extends}")
            else:
                self.diagnostics.add(
                    f"{path}: installed-package extends is unsupported: {extends}"
                )
        options = data.get("compilerOptions", {})
        if not isinstance(options, dict):
            self.diagnostics.add(f"{path}: compilerOptions must be an object")
            return base, inherited
        if "baseUrl" in options:
            value = options["baseUrl"]
            base = (
                normalize(posixpath.join(directory, value))
                if isinstance(value, str)
                else None
            )
            if base is None:
                self.diagnostics.add(f"{path}: invalid or outside-repository baseUrl")
                base = ""
        aliases = dict(inherited)
        if "paths" in options:
            aliases = {}
            mappings = options["paths"]
            if not isinstance(mappings, dict):
                self.diagnostics.add(f"{path}: paths must be an object")
                mappings = {}
            alias_base = base or directory
            for pattern, targets in mappings.items():
                if isinstance(targets, list):
                    aliases[pattern] = tuple(
                        candidate
                        for target in targets
                        if isinstance(target, str)
                        if (candidate := normalize(posixpath.join(alias_base, target)))
                        is not None
                    )
        return base, aliases

    def _file(self, stem: str) -> str | None:
        stem = normalize(stem)
        if stem is None:
            return None
        extension = PurePosixPath(stem).suffix
        substitutes = {
            ".js": (".ts", ".tsx", ".d.ts"),
            ".jsx": (".tsx", ".d.ts"),
            ".mjs": (".mts", ".d.mts"),
            ".cjs": (".cts", ".d.cts"),
        }
        candidates = [
            stem[: -len(extension)] + ext for ext in substitutes.get(extension, ())
        ]
        candidates.append(stem)
        if extension not in JS_EXTENSIONS | STYLE_EXTENSIONS:
            candidates.extend(stem + ext for ext in _EXTENSIONS)
        for candidate in candidates:
            if (
                candidate in self.files
                and PurePosixPath(candidate).suffix in JS_EXTENSIONS | STYLE_EXTENSIONS
            ):
                return candidate
        return None

    @staticmethod
    def _mapping(mapping, key: str) -> tuple[bool, list[str]]:
        def flatten(value):
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [target for child in value for target in flatten(child)]
            if isinstance(value, dict):
                return [target for child in value.values() for target in flatten(child)]
            return []

        if not isinstance(mapping, dict) or (
            key == "." and not any(k.startswith((".", "#")) for k in mapping)
        ):
            return (True, flatten(mapping)) if key == "." else (False, [])
        if key in mapping:
            return True, flatten(mapping[key])
        patterns = sorted(
            (p for p in mapping if p.count("*") == 1),
            key=lambda p: (len(p.split("*")[0]), len(p)),
            reverse=True,
        )
        for pattern in patterns:
            prefix, suffix = pattern.split("*")
            if (
                key.startswith(prefix)
                and key.endswith(suffix)
                and len(key) >= len(prefix) + len(suffix)
            ):
                middle = key[len(prefix) : len(key) - len(suffix) if suffix else None]
                return True, [
                    target.replace("*", middle) for target in flatten(mapping[pattern])
                ]
        return False, []

    def _package(
        self, directory: str, subpath: str = ".", *, private=False, seen=frozenset()
    ) -> set[str]:
        marker = (directory, subpath, private)
        if marker in seen:
            self.diagnostics.add(
                f"{directory}/package.json: cyclic package mapping {subpath}"
            )
            return set()
        seen = seen | {marker}
        data = self._json(posixpath.join(directory, "package.json"))
        field = "imports" if private else "exports"
        if field in data:
            _, targets = self._mapping(data[field], subpath)
        elif subpath == ".":
            targets = [
                data[key]
                for key in ("types", "typings", "module", "main")
                if isinstance(data.get(key), str)
            ]
            targets = targets or ["./index"]
        else:
            targets = [subpath]
        results = set()
        for target in targets:
            # Only local package targets are resolved. External dependencies
            # remain external and cannot trigger reads from node_modules.
            if private and target.startswith("#"):
                results.update(
                    self._package(directory, target, private=True, seen=seen)
                )
                continue
            if field in data and not target.startswith("./"):
                continue
            candidate = normalize(posixpath.join(directory, target))
            if candidate is None or (
                directory and not candidate.startswith(directory + "/")
            ):
                continue
            resolved = self._file(candidate) or self._file(candidate + "/index")
            if resolved:
                results.add(resolved)
        return results

    def _path(self, stem: str) -> set[str]:
        candidate = normalize(stem)
        if candidate is None:
            return set()
        direct = self._file(candidate)
        if direct:
            return {direct}
        if candidate + "/package.json" in self.files:
            return self._package(candidate)
        index = self._file(candidate + "/index")
        return {index} if index else set()

    def _javascript(self, path: str, specifier: str) -> tuple[set[str], bool]:
        directory = posixpath.dirname(path)
        if specifier.startswith("."):
            return self._path(posixpath.join(directory, specifier)), True
        if specifier.startswith("/"):
            return self._path(specifier.lstrip("/")), True
        base, aliases = self._config(path)
        matched, targets = self._mapping(aliases, specifier)
        # Alias target arrays are ordered fallback locations, not build variants.
        if matched:
            for target in targets:
                resolved = self._path(target)
                if resolved:
                    return resolved, True
            return set(), True
        if base:
            resolved = self._path(posixpath.join(base, specifier))
            if resolved:
                return resolved, True
        if specifier.startswith("#"):
            current = directory
            while True:
                if posixpath.join(current, "package.json") in self.files:
                    return self._package(current, specifier, private=True), True
                if not current:
                    return set(), True
                current = posixpath.dirname(current)
        parts = specifier.split("/")
        name = "/".join(parts[:2]) if specifier.startswith("@") else parts[0]
        if name in self.packages:
            directories = self.packages[name]
            if len(directories) != 1:
                self.diagnostics.add(f"{path}: ambiguous local package {name}")
                return set(), True
            subpath = "." + specifier[len(name) :]
            return self._package(directories[0], subpath), True
        return set(), False

    def analyze(self, path: str, source: bytes | None = None) -> ImportResult:
        try:
            parsed = self._parsed_sources.get(path) if source is None else None
            if parsed is None:
                parsed = parse(
                    path, self.snapshot.read(path) if source is None else source
                )
        except OSError as exc:
            result = ImportResult(0, (), (), (f"{path}: cannot read source: {exc}",))
            self.diagnostics.update(result.diagnostics)
            return result
        diagnostics = [f"{path}: {message}" for message in parsed.diagnostics]
        targets, external = [], []
        for reference in parsed.references:
            if path.endswith(".py"):
                resolved = self._python(path, reference)
                specifier = reference.specifier
                top = specifier.split(".")[0]
                local = bool(reference.level) or any(
                    stem in self._python_stems
                    for stem in (
                        top,
                        "src/" + top,
                        posixpath.join(posixpath.dirname(path), top),
                    )
                )
            else:
                specifier = reference.specifier
                resolved, local = self._javascript(path, specifier)
            targets.extend(sorted(resolved))
            if not resolved:
                if local:
                    diagnostics.append(
                        f"{path}: unresolved local import {specifier or '.'}"
                    )
                elif specifier:
                    external.append(specifier)
        self.diagnostics.update(diagnostics)
        return ImportResult(
            len(parsed.references),
            tuple(targets),
            tuple(external),
            tuple(
                sorted(
                    message
                    for message in self.diagnostics
                    if message.startswith(path + ": ")
                )
            ),
        )

    def graph(self) -> tuple[set[str], set[tuple[str, str]]]:
        nodes = {
            path
            for path in self.files
            if PurePosixPath(path).suffix
            in (SOURCE_EXTENSIONS - {".go", ".rs"}) | STYLE_EXTENSIONS
        }
        edges = set()
        for path in sorted(nodes):
            if PurePosixPath(path).suffix in SOURCE_EXTENSIONS - {".go", ".rs"}:
                edges.update((path, target) for target in self.analyze(path).targets)
        for language in (GoGraph, RustGraph):
            graph = language(self.snapshot, parsed_sources=self._parsed_sources)
            nodes.update(graph.results)
            edges.update(
                (node_id, target)
                for node_id, result in graph.results.items()
                for target in result.imports.targets
            )
            self.diagnostics.update(graph.diagnostics)
        return nodes, edges


__all__ = [
    "EXTRACTION_VERSION",
    "JS_EXTENSIONS",
    "ImportGraph",
    "ImportResult",
    "Snapshot",
    "report_diagnostics",
]
