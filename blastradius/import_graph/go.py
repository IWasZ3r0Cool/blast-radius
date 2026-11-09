# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Go syntax and package resolution over repository snapshots."""

import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .results import ImportResult, NodeAnalysis
from .snapshot import Snapshot, local_path, source_text


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


def _identifier_start(char: str) -> bool:
    return char == "_" or char.isalpha()


def _identifier_part(char: str) -> bool:
    return char == "_" or char.isalnum()


def _read_quoted(source: str, start: int, quote: str) -> tuple[str, int]:
    value = []
    index = start + 1
    while index < len(source):
        char = source[index]
        if quote == '"' and char == "\\" and index + 1 < len(source):
            value.append(source[index + 1])
            index += 2
            continue
        if char == quote:
            return "".join(value), index + 1
        value.append(char)
        index += 1
    return "".join(value), index


def _tokenize(source: str) -> list[_Token]:
    tokens = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline == -1 else newline + 1
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            index = len(source) if end == -1 else end + 2
            continue
        if char in {'"', "`"}:
            value, index = _read_quoted(source, index, char)
            tokens.append(_Token("string", value))
            continue
        if char == "'":
            _value, index = _read_quoted(source, index, char)
            continue
        if _identifier_start(char):
            end = index + 1
            while end < len(source) and _identifier_part(source[end]):
                end += 1
            tokens.append(_Token("identifier", source[index:end]))
            index = end
            continue
        tokens.append(_Token("punctuation", char))
        index += 1
    return tokens


def _scan_go_source(source: str) -> tuple[str | None, list[str]]:
    tokens = _tokenize(source)
    package_name = None
    imports = []
    seen = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.kind == "identifier"
            and token.value == "package"
            and index + 1 < len(tokens)
            and tokens[index + 1].kind == "identifier"
            and package_name is None
        ):
            package_name = tokens[index + 1].value
            index += 2
            continue
        if (
            token.kind != "identifier"
            or token.value != "import"
            or index + 1 >= len(tokens)
        ):
            index += 1
            continue

        index += 1
        if tokens[index].kind == "punctuation" and tokens[index].value == "(":
            depth = 1
            index += 1
            while index < len(tokens) and depth:
                current = tokens[index]
                if current.kind == "punctuation" and current.value == "(":
                    depth += 1
                elif current.kind == "punctuation" and current.value == ")":
                    depth -= 1
                elif (
                    current.kind == "string"
                    and depth == 1
                    and current.value not in seen
                ):
                    seen.add(current.value)
                    imports.append(current.value)
                index += 1
            continue

        for candidate in tokens[index : index + 3]:
            if candidate.kind == "string":
                if candidate.value not in seen:
                    seen.add(candidate.value)
                    imports.append(candidate.value)
                break
        index += 1
    return package_name, imports


@dataclass(frozen=True)
class GoParsed:
    package_name: str | None
    imports: tuple[str, ...]
    loc: int


def parse_go(source: bytes) -> GoParsed:
    text = source_text(source)
    name, imports = _scan_go_source(text)
    return GoParsed(name, tuple(imports), text.count("\n") + 1)


def _strip_comment(line: str) -> str:
    quoted = False
    raw = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if quoted and char == "\\":
            escaped = True
            continue
        if not raw and char == '"':
            quoted = not quoted
            continue
        if not quoted and char == "`":
            raw = not raw
            continue
        if not quoted and not raw and line.startswith("//", index):
            return line[:index]
    return line


def _directive_lines(source: str):
    mode = None
    lines = source.splitlines()
    for raw_line in lines:
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if mode:
            if line == ")":
                mode = None
                continue
            yield mode, line
            continue
        parts = line.split(None, 1)
        directive = parts[0]
        rest = parts[1].strip() if len(parts) > 1 else ""
        if rest == "(":
            mode = directive
            continue
        yield directive, rest


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "`"}:
        return value[1:-1]
    return value


def external_pkg_name(imp_path: str) -> str:
    """Condense an import path to its root module name for external packages."""
    parts = imp_path.split("/")
    if (
        parts[0] in {"github.com", "gitlab.com", "bitbucket.org", "gopkg.in"}
        and len(parts) >= 3
    ):
        return "/".join(parts[:3])
    if parts[0] in {"golang.org", "google.golang.org", "k8s.io"} and len(parts) >= 2:
        return "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts[:2])
    return parts[0]


@dataclass
class _Module:
    root: PurePosixPath
    name: str
    requires: set[str] = field(default_factory=set)
    replacements: dict[str, PurePosixPath | None] = field(default_factory=dict)


@dataclass
class _Workspace:
    root: PurePosixPath
    members: set[PurePosixPath] = field(default_factory=set)
    replacements: dict[str, PurePosixPath | None] = field(default_factory=dict)


def _contains(path, directory):
    return path == directory or directory in path.parents


def _matches(import_path, module_path):
    return import_path == module_path or import_path.startswith(module_path + "/")


class GoGraph:
    """Package dependencies resolved entirely against one repository snapshot."""

    def __init__(self, snapshot: Snapshot, *, parsed_sources: Mapping | None = None):
        self.diagnostics: set[str] = set()
        self.results: dict[str, NodeAnalysis] = {}
        self.modules: dict[PurePosixPath, _Module] = {}
        self.workspaces: list[_Workspace] = []
        parsed_sources = parsed_sources or {}
        for filename in sorted(snapshot.files):
            path = PurePosixPath(filename)
            if path.name not in {"go.mod", "go.work"}:
                continue
            try:
                directives = list(
                    _directive_lines(source_text(snapshot.read(filename)))
                )
            except OSError as exc:
                self.diagnostics.add(f"{filename}: cannot read manifest: {exc}")
                continue
            if path.name == "go.mod":
                name = next(
                    (
                        _unquote(v.split()[0])
                        for d, v in directives
                        if d == "module" and v.split()
                    ),
                    None,
                )
                if not name:
                    self.diagnostics.add(f"{filename}: missing Go module name")
                    continue
                manifest = _Module(path.parent, name)
                self.modules[path.parent] = manifest
            else:
                manifest = _Workspace(path.parent)
                self.workspaces.append(manifest)
            for directive, value in directives:
                try:
                    parts = shlex.split(value, comments=False)
                except ValueError:
                    self.diagnostics.add(
                        f"{filename}: invalid Go directive {directive}"
                    )
                    continue
                parts = [_unquote(part) for part in parts]
                if directive == "require" and parts and isinstance(manifest, _Module):
                    manifest.requires.add(parts[0])
                elif directive == "use" and parts and isinstance(manifest, _Workspace):
                    target = local_path(path.parent, parts[0], self.diagnostics)
                    if target is not None:
                        manifest.members.add(target)
                elif directive == "replace" and "=>" in parts:
                    arrow = parts.index("=>")
                    if arrow and arrow + 1 < len(parts):
                        replacement = parts[arrow + 1]
                        # Versioned remote replacements stay external and must
                        # not fall through to a matching local module.
                        manifest.replacements[parts[0]] = (
                            local_path(path.parent, replacement, self.diagnostics)
                            if replacement.startswith((".", "/", "\\"))
                            or ":" in replacement
                            else None
                        )
        for workspace in self.workspaces:
            for member in sorted(workspace.members):
                if member not in self.modules:
                    self.diagnostics.add(
                        f"{workspace.root}/go.work: missing workspace module {member}"
                    )
        packages: dict[str, list[tuple[str, GoParsed]]] = {}
        for path in sorted(snapshot.files):
            if not path.endswith(".go"):
                continue
            try:
                parsed = parsed_sources.get(path)
                if not isinstance(parsed, GoParsed):
                    parsed = parse_go(snapshot.read(path))
            except OSError as exc:
                self.diagnostics.add(f"{path}: cannot read source: {exc}")
                continue
            packages.setdefault(PurePosixPath(path).parent.as_posix(), []).append(
                (path, parsed)
            )
        self.packages = set(packages)
        for directory, files in packages.items():
            source_dir = PurePosixPath(directory)
            module = self._owner(source_dir)
            imports = sorted({item for _, parsed in files for item in parsed.imports})
            internal, external = [], []
            for item in imports:
                kind, target = self._resolve(item, source_dir, module)
                (internal if kind == "internal" else external).append(target)
            name = next((p.package_name for _, p in files if p.package_name), None)
            self.results[directory] = NodeAnalysis(
                directory,
                tuple(path for path, _ in files),
                name or (source_dir.name or "main"),
                sum(parsed.loc for _, parsed in files),
                ImportResult(len(imports), tuple(internal), tuple(external)),
            )

    def _owner(self, directory):
        return max(
            (m for m in self.modules.values() if _contains(directory, m.root)),
            key=lambda m: len(m.root.parts),
            default=None,
        )

    def _resolve(self, imported, source_dir, module):
        workspace = max(
            (w for w in self.workspaces if _contains(source_dir, w.root)),
            key=lambda w: len(w.root.parts),
            default=None,
        )
        if module is None or workspace is None or module.root not in workspace.members:
            workspace = None
        members = (
            [
                self.modules[root]
                for root in sorted(workspace.members)
                if root in self.modules
            ]
            if workspace
            else [module]
            if module
            else []
        )
        candidates: list[tuple[int, str, PurePosixPath | None]] = []
        if workspace:
            candidates.extend(
                (0, name, target)
                for name, target in workspace.replacements.items()
                if _matches(imported, name)
            )
        replacements: dict[str, set[PurePosixPath | None]] = {}
        for member in members:
            for name, target in member.replacements.items():
                replacements.setdefault(name, set()).add(target)
        for name, targets in replacements.items():
            if not _matches(imported, name):
                continue
            if workspace and name in workspace.replacements:
                continue
            if len(targets) > 1:
                self.diagnostics.add(
                    f"{source_dir}: conflicting local replacements for {name}"
                )
                candidates.append((1, name, None))
            else:
                candidates.append((1, name, next(iter(targets))))
        candidates.extend(
            (2, member.name, member.root)
            for member in members
            if _matches(imported, member.name)
        )
        if candidates:
            priority, prefix, target_root = min(
                candidates, key=lambda c: (c[0], -len(c[1]))
            )
            tied = {
                target
                for tier, name, target in candidates
                if tier == priority and name == prefix
            }
            if len(tied) > 1:
                self.diagnostics.add(f"{source_dir}: ambiguous Go module {prefix}")
                return "external", prefix
            if target_root is not None:
                suffix = imported[len(prefix) :].lstrip("/")
                target = local_path(target_root, suffix or ".", self.diagnostics)
                if target is not None and target.as_posix() in self.packages:
                    # A parent module never owns a nested module's packages.
                    owner = self._owner(target)
                    if owner is None or _contains(target_root, owner.root):
                        return "internal", target.as_posix()
            return "external", prefix
        requirements = [
            r for member in members for r in member.requires if _matches(imported, r)
        ]
        return "external", max(
            requirements, key=len
        ) if requirements else external_pkg_name(imported)
