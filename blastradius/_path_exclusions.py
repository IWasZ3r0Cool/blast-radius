# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
# Modified by Christopher Wilson.
"""Path component exclusions shared by detection and file collection."""

from __future__ import annotations

from collections.abc import Iterable

_SKIP_DIRS = frozenset(
    {
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
)


def is_excluded_path(parts: Iterable[str]) -> bool:
    """Match an excluded name or hidden component, including the filename."""
    return any(part.startswith(".") or part in _SKIP_DIRS for part in parts)
