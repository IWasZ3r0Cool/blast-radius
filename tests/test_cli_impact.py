# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import pytest

from blastradius.cli import _resolve_impact_file_id


@pytest.mark.parametrize("collection_type", [list, set], ids=["list", "set"])
@pytest.mark.parametrize(
    ("file_path", "candidate_ids", "expected"),
    [
        pytest.param(
            "service.py", ["pkg/service.py", "service.py"], "service.py", id="exact"
        ),
        pytest.param(
            "./pkg/service.py", ["pkg/service.py"], "pkg/service.py", id="dot-prefix"
        ),
        pytest.param(
            "service.py", ["pkg/service.py"], "pkg/service.py", id="short-path"
        ),
        pytest.param(
            "/missing/pkg/service.py",
            ["pkg/service.py"],
            "pkg/service.py",
            id="absolute",
        ),
        pytest.param(
            "missing/pkg/service.py",
            ["pkg/service.py"],
            "pkg/service.py",
            id="long-path",
        ),
        pytest.param("vice.py", ["service.py"], "service.py", id="partial-name"),
        pytest.param("missing.py", ["service.py"], None, id="not-found"),
        pytest.param("service.py", [], None, id="no-candidates"),
        pytest.param(
            "../../pkg/service.py",
            ["pkg/service.py"],
            "pkg/service.py",
            id="parent-prefix",
        ),
        pytest.param(
            ".config.py",
            ["pkg/config.py", ".config.py"],
            ".config.py",
            id="exact-dotfile",
        ),
        pytest.param(".config.py", ["config.py"], "config.py", id="stripped-dotfile"),
        pytest.param(
            r"pkg\service.py", ["pkg/service.py"], None, id="literal-separator"
        ),
        pytest.param(
            "pkg/./service.py", ["pkg/service.py"], None, id="literal-dot-segment"
        ),
    ],
)
def test_resolve_impact_file_id(file_path, candidate_ids, expected, collection_type):
    candidates = collection_type(candidate_ids)
    original = candidates.copy()

    assert _resolve_impact_file_id(file_path, candidates) == expected
    assert candidates == original


@pytest.mark.parametrize(
    ("file_path", "candidates", "expected"),
    [
        ("service.py", ["z/service.py", "a/service.py"], "z/service.py"),
        ("service.py", ["a/service.py", "z/service.py"], "a/service.py"),
        ("./service.py", ["pkg/service.py", "service.py"], "pkg/service.py"),
        ("./service.py", ["service.py", "pkg/service.py"], "service.py"),
    ],
)
def test_resolve_impact_file_id_keeps_first_suffix_match(
    file_path, candidates, expected
):
    original = candidates.copy()

    assert _resolve_impact_file_id(file_path, candidates) == expected
    assert candidates == original
