# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Presentation-independent dependency results for snapshot consumers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImportResult:
    count: int
    targets: tuple[str, ...]
    external: tuple[str, ...]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class NodeAnalysis:
    id: str
    sources: tuple[str, ...]
    name: str
    loc: int
    imports: ImportResult
