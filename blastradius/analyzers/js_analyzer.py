# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""JavaScript / TypeScript / Vue repository analyzer (syntax-based imports)."""

import json
import re
from pathlib import Path

from blastradius.import_graph import (
    JS_EXTENSIONS,
    ImportGraph,
    Snapshot,
    report_diagnostics,
)

from .base import dir_group

CONFIG_STEMS = {
    "config",
    "settings",
    "constants",
    "env",
    "configuration",
    "conf",
    "vite.config",
    "webpack.config",
    "babel.config",
    "jest.config",
    "vitest.config",
    "tailwind.config",
    "next.config",
    "nuxt.config",
    "svelte.config",
    "astro.config",
    "rollup.config",
    "esbuild.config",
}

# Path-based heuristics
_ROUTE_DIRS = {"pages", "routes", "views", "screens"}
_NEXT_APP_FILES = {
    "page",
    "layout",
    "loading",
    "error",
    "not-found",
    "template",
    "route",
}
_STORE_DIRS = {
    "store",
    "stores",
    "state",
    "redux",
    "slices",
    "atoms",
    "contexts",
    "context",
}
_STORE_STEMS = {
    "store",
    "slice",
    "reducer",
    "context",
    "atom",
    "provider",
    "actions",
    "mutations",
}

# Frameworks detected by package.json dependency names
FRAMEWORK_SIGNALS = [
    ("next", "next"),
    ("nuxt", "nuxt"),
    ("@sveltejs/kit", "sveltekit"),
    ("svelte", "svelte"),
    ("@angular/core", "angular"),
    ("gatsby", "gatsby"),
    ("remix", "remix"),
    ("astro", "astro"),
    ("react", "react"),
    ("vue", "vue"),
]

# ── Component / hook / store detection regexes ────────────────────────────────
# JSX return: return ( <... or return <...
_JSX_RETURN_RE = re.compile(r"return\s*\(?[\s\n]*<[A-Za-z/]", re.MULTILINE)
# PascalCase JSX element usage (strong signal of component file).
# Negative lookbehind (?<!\w) excludes TypeScript generics like Promise<Response>.
_JSX_PASCAL_RE = re.compile(r"(?<!\w)<[A-Z][A-Za-z]+[\s/>]")
# export function/const useXxx or export default function useXxx
_HOOK_EXPORT_RE = re.compile(r"export\s+(?:default\s+)?(?:const|function)\s+use[A-Z]")
# Context API
_CONTEXT_RE = re.compile(r"createContext\s*[(<]")
# State management: Redux, Zustand, Jotai, Svelte stores
_STORE_RE = re.compile(
    r"createSlice\s*\(|createStore\s*\(|atom\s*\(|writable\s*\("
    r"|readable\s*\(|create\s*\(\s*\((?:set|get)\)"
)


def read_package_json(root: Path):
    """Returns (all_dep_names, detected_framework, package_manager)."""
    pkg_file = root / "package.json"
    if not pkg_file.exists():
        return set(), None, None

    try:
        pkg = json.loads(pkg_file.read_text(errors="replace"))
    except (json.JSONDecodeError, OSError):
        return set(), None, None

    all_deps = set()
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        all_deps.update(pkg.get(key, {}).keys())

    framework = next(
        (fw for signal, fw in FRAMEWORK_SIGNALS if signal in all_deps),
        None,
    )

    if (root / "pnpm-lock.yaml").exists():
        pm = "pnpm"
    elif (root / "yarn.lock").exists():
        pm = "yarn"
    elif (root / "package-lock.json").exists():
        pm = "npm"
    else:
        pm = None

    return all_deps, framework, pm


def detect_language(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".vue":
        return "vue"
    if suffix in {".ts", ".tsx", ".mts", ".cts"}:
        return "typescript"
    return "javascript"


def node_type(path: Path, source: str = "") -> str:
    """Classify a file into a semantic node type using path + content heuristics."""
    stem = path.stem.lower()
    ext = path.suffix.lower()
    parts_lower = [p.lower() for p in path.parts]

    # ── Config files ──────────────────────────────────────────────────────────
    if stem in CONFIG_STEMS:
        return "config"
    for name in CONFIG_STEMS:
        if stem.endswith(f".{name}"):
            return "config"

    # ── Vue SFC → always a component ─────────────────────────────────────────
    if ext == ".vue":
        return "component"

    # ── Route detection (by directory convention) ────────────────────────────
    if any(p in _ROUTE_DIRS for p in parts_lower):
        return "route"
    # Next.js app router special file names
    if "app" in parts_lower and ext in {".jsx", ".tsx"} and stem in _NEXT_APP_FILES:
        return "route"

    # ── Store / state management detection ───────────────────────────────────
    if any(p in _STORE_DIRS for p in parts_lower):
        return "store"
    if any(kw in stem for kw in _STORE_STEMS):
        return "store"
    if source and (_CONTEXT_RE.search(source) or _STORE_RE.search(source)):
        return "store"

    # ── Hook detection ────────────────────────────────────────────────────────
    # Filename starts with "use" + capital letter (e.g., useAuth.ts)
    if stem.startswith("use") and len(stem) > 3 and stem[3].isupper():
        return "hook"
    if source and _HOOK_EXPORT_RE.search(source):
        return "hook"

    # ── Component detection (JSX files or files containing JSX) ─────────────
    if ext in {".jsx", ".tsx"}:
        return "component"
    # .js/.ts files that contain JSX syntax
    if source and (_JSX_RETURN_RE.search(source) or _JSX_PASCAL_RE.search(source)):
        return "component"

    return "module"


def package_name(mod: str) -> str:
    """Extract the top-level package name (handles @scope/pkg paths)."""
    if mod.startswith("@"):
        parts = mod.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else mod
    return mod.split("/")[0]


def analyze(root: Path, group_map: dict):
    """
    Returns (nodes, external_nodes, links_map, meta).
    links_map keys are (source_rel, target_rel) tuples.
    """
    snapshot = Snapshot.directory(root)
    js_files = [
        root / path
        for path in sorted(snapshot.files)
        if Path(path).suffix in JS_EXTENSIONS
    ]

    if not js_files:
        return [], [], {}, {"total_files": 0, "total_loc": 0}

    _ext_deps, framework, package_manager = read_package_json(root)
    imports = ImportGraph(snapshot)

    nodes = []
    links_map = {}
    external_nodes = {}
    total_loc = 0

    for f in js_files:
        rel = str(f.relative_to(root))
        try:
            source_bytes = f.read_bytes()
            source = source_bytes.decode("utf-8-sig", errors="replace")
        except OSError:
            continue

        loc = source.count("\n") + 1
        total_loc += loc

        lang = detect_language(f)
        ntype = node_type(f.relative_to(root), source)

        result = imports.analyze(rel.replace("\\", "/"), source_bytes)

        nodes.append(
            {
                "id": rel,
                "type": ntype,
                "language": lang,
                "framework": framework,
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
            pkg = package_name(mod)
            if pkg not in external_nodes:
                external_nodes[pkg] = {
                    "id": pkg,
                    "type": "import",
                    "language": lang,
                    "size": 40,
                    "loc": 0,
                    "group": 9000,
                    "imports": 0,
                }
            key = (rel, pkg)
            links_map[key] = links_map.get(key, 0) + 1

    report_diagnostics(imports.diagnostics)

    meta = {
        "total_files": len(js_files),
        "total_loc": total_loc,
    }
    if framework:
        meta["framework"] = framework
    if package_manager:
        meta["packageManager"] = package_manager

    return nodes, list(external_nodes.values()), links_map, meta
