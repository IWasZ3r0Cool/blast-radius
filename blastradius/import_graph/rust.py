# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Rust syntax and module resolution over repository snapshots."""

import sys
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .results import ImportResult, NodeAnalysis
from .snapshot import Snapshot, local_path, source_text

STDLIB_CRATES = {
    "std",
    "core",
    "alloc",
    "proc_macro",
    "test",
    "crate",
    "super",
    "self",
}


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


@dataclass(frozen=True)
class _ModDecl:
    name: str
    inline_scope: tuple[str, ...]
    path_override: str | None = None


@dataclass(frozen=True)
class RustParsed:
    uses: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()
    extern_crates: tuple[tuple[str, tuple[str, ...]], ...] = ()
    mods: tuple[_ModDecl, ...] = ()
    loc: int = 0


@dataclass
class _CargoPackage:
    root: PurePosixPath
    data: dict
    name: str
    lib_name: str
    targets: list[_CrateTarget] = field(default_factory=list)
    local_dependencies: dict[str, _CargoPackage] = field(default_factory=dict)


@dataclass
class _CrateTarget:
    root_file: PurePosixPath | None
    crate_name: str
    kind: str
    package: _CargoPackage | None
    modules: dict[tuple[str, ...], list[str]] = field(default_factory=dict)
    declared_modules: dict[tuple[str, ...], list[str]] = field(default_factory=dict)
    file_modules: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)


def _identifier_start(char: str) -> bool:
    return char == "_" or char.isalpha()


def _identifier_part(char: str) -> bool:
    return char == "_" or char.isalnum()


def _read_quoted(source: str, start: int, quote: str) -> tuple[str, int]:
    value = []
    index = start + 1
    while index < len(source):
        char = source[index]
        if char == "\\" and index + 1 < len(source):
            value.append(source[index + 1])
            index += 2
            continue
        if char == quote:
            return "".join(value), index + 1
        value.append(char)
        index += 1
    return "".join(value), index


def _read_raw_string(source: str, start: int) -> tuple[str, int] | None:
    index = start
    if source.startswith("br", index):
        index += 2
    elif source.startswith("r", index):
        index += 1
    else:
        return None
    hashes = 0
    while index < len(source) and source[index] == "#":
        hashes += 1
        index += 1
    if index >= len(source) or source[index] != '"':
        return None
    content_start = index + 1
    delimiter = '"' + ("#" * hashes)
    end = source.find(delimiter, content_start)
    if end == -1:
        return source[content_start:], len(source)
    return source[content_start:end], end + len(delimiter)


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
            depth = 1
            index += 2
            while index < len(source) and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            continue
        raw = _read_raw_string(source, index)
        if raw is not None:
            value, index = raw
            tokens.append(_Token("string", value))
            continue
        if char == '"' or (char == "b" and source[index : index + 2] == 'b"'):
            quote_index = index if char == '"' else index + 1
            value, index = _read_quoted(source, quote_index, '"')
            tokens.append(_Token("string", value))
            continue
        if char == "'":
            closing = source.find("'", index + 1)
            if closing != -1 and closing - index <= 6:
                index = closing + 1
                continue
        if (
            source.startswith("r#", index)
            and index + 2 < len(source)
            and _identifier_start(source[index + 2])
        ):
            end = index + 3
            while end < len(source) and _identifier_part(source[end]):
                end += 1
            tokens.append(_Token("identifier", source[index + 2 : end]))
            index = end
            continue
        if _identifier_start(char):
            end = index + 1
            while end < len(source) and _identifier_part(source[end]):
                end += 1
            tokens.append(_Token("identifier", source[index:end]))
            index = end
            continue
        if source.startswith("::", index):
            tokens.append(_Token("punctuation", "::"))
            index += 2
            continue
        tokens.append(_Token("punctuation", char))
        index += 1
    return tokens


def _split_top_level(tokens: list[_Token]) -> list[list[_Token]]:
    groups = []
    start = 0
    depth = 0
    for index, token in enumerate(tokens):
        if token.value == "{":
            depth += 1
        elif token.value == "}":
            depth -= 1
        elif token.value == "," and depth == 0:
            groups.append(tokens[start:index])
            start = index + 1
    groups.append(tokens[start:])
    return [group for group in groups if group]


def _path_parts(tokens: list[_Token]) -> tuple[str, ...]:
    parts = []
    for token in tokens:
        if token.value == "as":
            break
        if token.kind == "identifier":
            parts.append(token.value)
    return tuple(parts)


def _expand_use_tree(
    tokens: list[_Token], prefix: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    while tokens and tokens[0].value == "::":
        tokens = tokens[1:]
    brace_index = None
    depth = 0
    close_index = None
    for index, token in enumerate(tokens):
        if token.value == "{":
            if depth == 0 and brace_index is None:
                brace_index = index
            depth += 1
        elif token.value == "}":
            depth -= 1
            if depth == 0 and brace_index is not None:
                close_index = index
                break
    if brace_index is None:
        parts = _path_parts(tokens)
        if parts == ("self",) and prefix:
            return [prefix]
        return [prefix + parts] if parts or prefix else []

    base = prefix + _path_parts(tokens[:brace_index])
    inside = tokens[brace_index + 1 : close_index]
    expanded = []
    for group in _split_top_level(inside):
        expanded.extend(_expand_use_tree(group, base))
    return expanded


def _find_statement_end(tokens: list[_Token], start: int) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        value = tokens[index].value
        if value == "{":
            depth += 1
        elif value == "}":
            depth -= 1
        elif value == ";" and depth == 0:
            return index
    return len(tokens)


def _scan_source(source: str) -> RustParsed:
    tokens = _tokenize(source)
    uses, extern_crates, mods = [], [], []
    brace_depth = 0
    inline_stack: list[tuple[int, str]] = []
    inline_openings: dict[int, str] = {}
    pending_path: str | None = None
    index = 0

    while index < len(tokens):
        token = tokens[index]
        value = token.value
        scope = tuple(name for _depth, name in inline_stack)

        if value == "#" and index + 4 < len(tokens) and tokens[index + 1].value == "[":
            end = index + 2
            attr_depth = 1
            while end < len(tokens) and attr_depth:
                if tokens[end].value == "[":
                    attr_depth += 1
                elif tokens[end].value == "]":
                    attr_depth -= 1
                end += 1
            attribute = tokens[index + 2 : end - 1]
            for attr_index in range(len(attribute) - 2):
                if (
                    attribute[attr_index].value == "path"
                    and attribute[attr_index + 1].value == "="
                    and attribute[attr_index + 2].kind == "string"
                ):
                    pending_path = attribute[attr_index + 2].value
                    break
            index = end
            continue

        if value == "use":
            end = _find_statement_end(tokens, index + 1)
            for path in _expand_use_tree(tokens[index + 1 : end]):
                if path:
                    uses.append((path, scope))
            pending_path = None
            index = end + 1
            continue

        if (
            value == "extern"
            and index + 2 < len(tokens)
            and tokens[index + 1].value == "crate"
            and tokens[index + 2].kind == "identifier"
        ):
            extern_crates.append((tokens[index + 2].value, scope))

        if value == "mod" and index + 1 < len(tokens):
            name_token = tokens[index + 1]
            if name_token.kind == "identifier":
                next_index = index + 2
                if next_index < len(tokens) and tokens[next_index].value == ";":
                    mods.append(_ModDecl(name_token.value, scope, pending_path))
                elif next_index < len(tokens) and tokens[next_index].value == "{":
                    inline_openings[next_index] = name_token.value
                pending_path = None

        if value == "{":
            brace_depth += 1
            if index in inline_openings:
                inline_stack.append((brace_depth, inline_openings[index]))
            pending_path = None
        elif value == "}":
            if inline_stack and inline_stack[-1][0] == brace_depth:
                inline_stack.pop()
            brace_depth = max(0, brace_depth - 1)
            pending_path = None
        elif value == ";":
            pending_path = None

        index += 1
    return RustParsed(
        tuple(uses), tuple(extern_crates), tuple(mods), source.count("\n") + 1
    )


def parse_rust(source: bytes) -> RustParsed:
    return _scan_source(source_text(source))


def _relative(path: PurePosixPath, root: PurePosixPath) -> str:
    return path.relative_to(root).as_posix()


def read_manifest(snapshot: Snapshot, filename: str, diagnostics: set[str]) -> dict:
    try:
        return tomllib.loads(source_text(snapshot.read(filename)))
    except (ValueError, OSError) as exc:
        diagnostics.add(f"{filename}: invalid Cargo manifest: {exc}")
        return {}


def _discover_packages(snapshot: Snapshot, diagnostics: set[str]):
    manifests, packages = [], []
    for filename in sorted(snapshot.files):
        manifest = PurePosixPath(filename)
        if manifest.name != "Cargo.toml":
            continue
        data = read_manifest(snapshot, filename, diagnostics)
        manifests.append((manifest.parent, data))
        package_data = data.get("package")
        if not isinstance(package_data, dict) or not package_data.get("name"):
            continue
        name = str(package_data["name"])
        lib = data.get("lib", {}) if isinstance(data.get("lib"), dict) else {}
        lib_name = str(lib.get("name", name.replace("-", "_")))
        package = _CargoPackage(manifest.parent, data, name, lib_name)
        raw_targets: list[tuple[PurePosixPath, str, str]] = []

        def add(value, target_name, kind, package=package, raw_targets=raw_targets):
            path = local_path(package.root, str(value), diagnostics)
            if path is not None and path.as_posix() in snapshot.files:
                raw_targets.append((path, target_name, kind))

        if lib.get("path"):
            add(lib["path"], str(lib.get("name", "")), "lib")
        elif package_data.get("autolib", True):
            add("src/lib.rs", lib_name, "lib")
        for table, kind in (
            ("bin", "bin"),
            ("example", "example"),
            ("test", "test"),
            ("bench", "bench"),
        ):
            for spec in data.get(table, []):
                if isinstance(spec, dict) and spec.get("path"):
                    add(spec["path"], str(spec.get("name", "")), kind)
        build = package_data.get("build", "build.rs")
        if build is not False:
            add(build if isinstance(build, str) else "build.rs", name, "build")
        if package_data.get("autobins", True):
            add("src/main.rs", name, "bin")
        for directory, flag, kind in (
            ("src/bin", "autobins", "bin"),
            ("examples", "autoexamples", "example"),
            ("tests", "autotests", "test"),
            ("benches", "autobenches", "bench"),
        ):
            if not package_data.get(flag, True):
                continue
            base = package.root / directory
            for filename in sorted(snapshot.files):
                path = PurePosixPath(filename)
                if path.suffix != ".rs":
                    continue
                if path.parent == base or (
                    path.name == "main.rs" and path.parent.parent == base
                ):
                    raw_targets.append(
                        (
                            path,
                            path.stem if path.parent == base else path.parent.name,
                            kind,
                        )
                    )
        seen = set()
        for path, target_name, kind in raw_targets:
            if path in seen:
                continue
            seen.add(path)
            package.targets.append(
                _CrateTarget(
                    path,
                    target_name or (lib_name if kind == "lib" else path.stem),
                    kind,
                    package,
                )
            )
        packages.append(package)
    return packages, manifests


def _dependency_specs(data: dict):
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        table = data.get(section, {})
        if isinstance(table, dict):
            yield from table.items()
    targets = data.get("target", {})
    if isinstance(targets, dict):
        for target_data in targets.values():
            if not isinstance(target_data, dict):
                continue
            for section in ("dependencies", "dev-dependencies", "build-dependencies"):
                table = target_data.get(section, {})
                if isinstance(table, dict):
                    yield from table.items()


def _workspace_dependency(
    package: _CargoPackage, alias: str, manifests: list[tuple[PurePosixPath, dict]]
) -> tuple[object, PurePosixPath] | None:
    candidates = []
    for manifest_root, data in manifests:
        try:
            package.root.relative_to(manifest_root)
        except ValueError:
            continue
        workspace = data.get("workspace", {})
        dependencies = (
            workspace.get("dependencies", {}) if isinstance(workspace, dict) else {}
        )
        if alias in dependencies:
            candidates.append(
                (len(manifest_root.parts), dependencies[alias], manifest_root)
            )
    if not candidates:
        return None
    _depth, spec, manifest_root = max(candidates, key=lambda item: item[0])
    return spec, manifest_root


def _connect_local_dependencies(
    packages: list[_CargoPackage],
    manifests: list[tuple[PurePosixPath, dict]],
    diagnostics: set[str],
) -> None:
    by_root = {package.root: package for package in packages}
    for package in packages:
        for raw_alias, raw_spec in _dependency_specs(package.data):
            alias = str(raw_alias).replace("-", "_")
            spec = raw_spec
            spec_root = package.root
            if isinstance(spec, dict) and spec.get("workspace") is True:
                inherited = _workspace_dependency(package, str(raw_alias), manifests)
                if inherited is not None:
                    spec, spec_root = inherited
            target = None
            if isinstance(spec, dict) and spec.get("path"):
                target_path = local_path(spec_root, str(spec["path"]), diagnostics)
                target = by_root.get(target_path)
            if target is not None:
                package.local_dependencies[alias] = target


def _module_parts(path: PurePosixPath, base_dir: PurePosixPath) -> tuple[str, ...]:
    relative = path.relative_to(base_dir)
    if path.name == "mod.rs":
        return relative.parent.parts
    return relative.with_suffix("").parts


def _module_candidate(
    source: PurePosixPath,
    declaration: _ModDecl,
    is_crate_root: bool,
    all_files: set[str],
    root: PurePosixPath,
    diagnostics: set[str] | None = None,
) -> str | None:
    if declaration.path_override:
        if declaration.inline_scope:
            base = (
                source.parent
                if is_crate_root or source.name == "mod.rs"
                else source.parent / source.stem
            )
            base = base.joinpath(*declaration.inline_scope)
        else:
            base = source.parent
        checked = local_path(
            base,
            declaration.path_override,
            diagnostics if diagnostics is not None else set(),
        )
        candidates = [checked] if checked is not None else []
    else:
        base = (
            source.parent
            if is_crate_root or source.name == "mod.rs"
            else source.parent / source.stem
        )
        base = base.joinpath(*declaration.inline_scope)
        candidates = [
            base / f"{declaration.name}.rs",
            base / declaration.name / "mod.rs",
        ]
    for candidate in candidates:
        checked = local_path(
            PurePosixPath("."),
            candidate.as_posix(),
            diagnostics if diagnostics is not None else set(),
        )
        if checked is None:
            continue
        relative = checked.as_posix()
        if relative in all_files:
            return relative
    return None


def _add_module(
    target: _CrateTarget,
    module: tuple[str, ...],
    relative: str,
    *,
    declared: bool = False,
) -> bool:
    if declared:
        declared_candidates = target.declared_modules.setdefault(module, [])
        if relative not in declared_candidates:
            declared_candidates.append(relative)
    candidates = target.modules.setdefault(module, [])
    if relative in candidates:
        return False
    candidates.append(relative)
    paths = target.file_modules.setdefault(relative, [])
    if module not in paths:
        paths.append(module)
    return True


def _build_target_modules(
    target: _CrateTarget,
    package_files: list[PurePosixPath],
    other_targets: list[_CrateTarget],
    all_files: set[str],
    root: PurePosixPath,
    source_items: dict[str, RustParsed],
    diagnostics: set[str],
) -> None:
    if target.root_file is None:
        base_dir = target.package.root if target.package else root
    else:
        base_dir = target.root_file.parent
        _add_module(target, (), _relative(target.root_file, root))

    excluded_roots = {
        other.root_file
        for other in other_targets
        if other is not target and other.root_file is not None
    }
    excluded_dirs = {
        other.root_file.parent
        for other in other_targets
        if other is not target
        and other.root_file is not None
        and other.root_file.name == "main.rs"
        and other.root_file.parent != base_dir
    }
    for path in package_files:
        resolved = path
        if target.root_file is not None and resolved == target.root_file:
            continue
        if resolved in excluded_roots:
            continue
        if any(directory in resolved.parents for directory in excluded_dirs):
            continue
        try:
            module = _module_parts(path, base_dir)
        except ValueError:
            continue
        _add_module(target, module, _relative(path, root))

    pending = deque(
        (relative, module, frozenset({relative}))
        for relative, modules in target.file_modules.items()
        for module in modules
    )
    while pending:
        relative, source_module, ancestors = pending.popleft()
        source = root / relative
        items = source_items.get(relative, RustParsed())
        for declaration in items.mods:
            candidate = _module_candidate(
                source,
                declaration,
                target.root_file is not None and source == target.root_file,
                all_files,
                root,
                diagnostics,
            )
            if candidate is None:
                continue
            if candidate in ancestors:
                diagnostics.add(
                    f"{relative}: cyclic Rust module declaration {declaration.name}"
                )
                continue
            logical = source_module + declaration.inline_scope + (declaration.name,)
            if _add_module(target, logical, candidate, declared=True):
                pending.append((candidate, logical, ancestors | {candidate}))


def _discover_targets(
    root: PurePosixPath,
    files: list[PurePosixPath],
    snapshot: Snapshot,
    source_items: dict[str, RustParsed],
    diagnostics: set[str],
) -> list[_CrateTarget]:
    packages, manifests = _discover_packages(snapshot, diagnostics)
    _connect_local_dependencies(packages, manifests, diagnostics)
    targets = [target for package in packages for target in package.targets]

    if not targets:
        roots = [path for path in files if path.name in {"lib.rs", "main.rs"}]
        if roots:
            targets = [_CrateTarget(path, path.stem, "loose", None) for path in roots]
        else:
            targets = [_CrateTarget(None, "", "loose", None)]

    package_roots = sorted(
        (package.root for package in packages),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    package_files: dict[PurePosixPath | None, list[PurePosixPath]] = {
        package.root: [] for package in packages
    }
    package_files[None] = []
    for path in files:
        owner = next(
            (candidate for candidate in package_roots if candidate in path.parents),
            None,
        )
        package_files.setdefault(owner, []).append(path)

    all_files = {_relative(path, root) for path in files}
    for target in targets:
        owner = target.package.root if target.package else None
        siblings = target.package.targets if target.package else targets
        _build_target_modules(
            target,
            package_files.get(owner, files),
            siblings,
            all_files,
            root,
            source_items,
            diagnostics,
        )
    return targets


def _longest_modules(
    target: _CrateTarget, parts: tuple[str, ...], minimum: int
) -> list[str]:
    for length in range(len(parts), minimum - 1, -1):
        prefix = parts[:length]
        # Explicit declarations replace filesystem guesses, but retain every
        # statically visible conditional variant at the same logical path.
        candidates = target.declared_modules.get(prefix) or target.modules.get(prefix)
        if candidates:
            return candidates
    return []


def _library_target(package: _CargoPackage) -> _CrateTarget | None:
    return next((target for target in package.targets if target.kind == "lib"), None)


def _resolve_use(
    parts: tuple[str, ...],
    source_module: tuple[str, ...],
    target: _CrateTarget,
) -> list[tuple[str, str]]:
    if not parts:
        return []
    first = parts[0]
    if first == "crate":
        return [("internal", path) for path in _longest_modules(target, parts[1:], 0)]
    if first in {"self", "super"}:
        base = list(source_module)
        index = 0
        if first == "self":
            index = 1
        else:
            while index < len(parts) and parts[index] == "super":
                if base:
                    base.pop()
                index += 1
        return [
            ("internal", path)
            for path in _longest_modules(target, tuple(base) + parts[index:], 0)
        ]

    for base in (source_module, ()):
        resolved = _longest_modules(target, base + parts, len(base) + 1)
        if resolved:
            return [("internal", path) for path in resolved]

    package = target.package
    dependency = package.local_dependencies.get(first) if package else None
    if (
        dependency is None
        and package
        and first == package.lib_name
        and target.kind != "lib"
    ):
        dependency = package
    if dependency is not None:
        library = _library_target(dependency)
        if library is not None:
            resolved = _longest_modules(library, parts[1:], 0)
            if resolved:
                return [("internal", path) for path in resolved]

    if first not in STDLIB_CRATES:
        return [("external", first)]
    return []


class RustGraph:
    """Rust file dependencies with all statically visible module variants."""

    def __init__(self, snapshot: Snapshot, *, parsed_sources: Mapping | None = None):
        self.diagnostics: set[str] = set()
        self.results: dict[str, NodeAnalysis] = {}
        parsed_sources = parsed_sources or {}
        items_by_file = {}
        for filename in sorted(snapshot.files):
            if not filename.endswith(".rs"):
                continue
            try:
                parsed = parsed_sources.get(filename)
                if not isinstance(parsed, RustParsed):
                    parsed = parse_rust(snapshot.read(filename))
                items_by_file[filename] = parsed
            except OSError as exc:
                self.diagnostics.add(f"{filename}: cannot read source: {exc}")
        root = PurePosixPath(".")
        files = [PurePosixPath(path) for path in items_by_file]
        targets = _discover_targets(
            root, files, snapshot, items_by_file, self.diagnostics
        )
        all_files = set(items_by_file)
        contexts_by_file = {}
        for target in targets:
            for filename in target.file_modules:
                contexts_by_file.setdefault(filename, []).append(target)
        for filename, items in items_by_file.items():
            path = PurePosixPath(filename)
            contexts = contexts_by_file.get(filename, [])
            resolved = []
            for declaration in items.mods:
                for context in contexts or [None]:
                    is_root = context is not None and path == context.root_file
                    target_path = _module_candidate(
                        path, declaration, is_root, all_files, root, self.diagnostics
                    )
                    if target_path:
                        resolved.append(("internal", target_path))
            for context in contexts:
                for use_path, inline_scope in items.uses:
                    for module in context.file_modules.get(filename, [()]):
                        resolved.extend(
                            _resolve_use(use_path, module + inline_scope, context)
                        )
                for crate, inline_scope in items.extern_crates:
                    for module in context.file_modules.get(filename, [()]):
                        resolved.extend(
                            _resolve_use((crate,), module + inline_scope, context)
                        )
            deduped = list(
                dict.fromkeys(
                    item for item in resolved if item != ("internal", filename)
                )
            )
            self.results[filename] = NodeAnalysis(
                filename,
                (filename,),
                path.stem,
                items.loc,
                ImportResult(
                    len(deduped),
                    tuple(p for k, p in deduped if k == "internal"),
                    tuple(p for k, p in deduped if k == "external"),
                ),
            )


def resolve_crate_use(use_path: str, all_files: set):
    """Compatibility helper for resolving simple crate-relative paths."""
    if not use_path.startswith("crate::"):
        return None
    parts = tuple(use_path[7:].split("::"))
    for length in range(len(parts), 0, -1):
        relative = "/".join(parts[:length])
        for prefix in ("src", ""):
            base = f"{prefix}/{relative}" if prefix else relative
            for candidate in (base + ".rs", base + "/mod.rs"):
                if candidate in all_files:
                    return candidate
    return None
