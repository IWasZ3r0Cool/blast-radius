# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Repository files independent of their working-directory or Git storage."""

import os
import posixpath
import re
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

SKIP_DIRS = {
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".git",
    "dist",
    "build",
    ".next",
    "out",
    "coverage",
    ".turbo",
    ".cache",
    "target",
    "vendor",
    ".expo",
}
JS_EXTENSIONS = frozenset(
    {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts", ".vue"}
)
STYLE_EXTENSIONS = frozenset({".css", ".scss", ".sass", ".less", ".styl"})
SOURCE_EXTENSIONS = JS_EXTENSIONS | {".py"}


def ignore_patterns(content: str) -> list[re.Pattern]:
    patterns = []
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pattern = (
                re.escape(line)
                .replace(r"\*\*", ".*")
                .replace(r"\*", "[^/]*")
                .replace(r"\?", ".")
            )
            patterns.append(re.compile(pattern))
    return patterns


def skipped(path: str) -> bool:
    return any(
        part.startswith(".") or part in SKIP_DIRS for part in PurePosixPath(path).parts
    )


def normalize(path: str) -> str | None:
    result = posixpath.normpath(path)
    if result.startswith(("/", "../")) or result == "..":
        return None
    return result


class Snapshot:
    """An immutable inventory and byte reader; paths always use POSIX separators."""

    def __init__(self, paths: Iterable[str], read_bytes: Callable[[str], bytes]):
        self.paths = frozenset(paths)
        self.read_bytes = read_bytes
        patterns = ignore_patterns(
            self.read(".gitignore").decode("utf-8", errors="replace")
        )
        self.files = frozenset(
            path
            for path in self.paths
            if not skipped(path)
            and not any(pattern.search(path) for pattern in patterns)
        )

    def read(self, path: str) -> bytes:
        if path not in self.paths:
            return b""
        return self.read_bytes(path)

    @classmethod
    def directory(cls, root: Path) -> Snapshot:
        root = root.resolve()
        paths = []
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if not skipped(d))
            for name in sorted(files):
                path = Path(directory) / name
                # Never follow a source/config symlink outside the repository.
                try:
                    path.resolve().relative_to(root)
                except (ValueError, RuntimeError, OSError):
                    continue
                paths.append(path.relative_to(root).as_posix())
        return cls(paths, lambda path: (root / path).read_bytes())
