# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from blastradius.symbol_impact import python_usage, symbol_impact


def test_aliases_and_caller_locations():
    result = python_usage(
        "from auth import validate as check\ndef route():\n    return check()\n",
        "api.py",
        "auth.py",
        "validate",
    )
    assert result["usage"] == "referenced"
    assert result["references"] == [{"line": 3, "column": 11, "symbol": "route"}]


def test_comments_strings_and_other_exports_are_not_usage():
    result = python_usage(
        'from auth import validate\n# validate()\nmessage = "validate"\n',
        "api.py",
        "auth.py",
        "validate",
    )
    assert result["imports_symbol"] is True
    assert result["usage"] == "unused"
    assert (
        python_usage("from auth import other\n", "api.py", "auth.py", "validate")[
            "imports_symbol"
        ]
        is False
    )


def test_relative_and_module_aliases():
    for source in [
        "from .auth import validate\nvalidate()\n",
        "import pkg.auth as a\na.validate()\n",
        "import pkg.auth\npkg.auth.validate()\n",
    ]:
        assert (
            python_usage(source, "pkg/api.py", "pkg/auth.py", "validate")["usage"]
            == "referenced"
        )


def test_shadowing_and_wildcards_are_unknown():
    for source in [
        "from auth import *\nvalidate()\n",
        "from auth import validate\ndef route(validate):\n    validate()\n",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from auth import validate\n",
    ]:
        assert (
            python_usage(source, "api.py", "auth.py", "validate")["usage"] == "unknown"
        )


def test_filtered_graph_cycle_and_unknown_source(tmp_path):
    (tmp_path / "used.py").write_text("from auth import validate\nvalidate()\n")
    (tmp_path / "unused.py").write_text("from auth import validate\n")

    class Store:
        def importers_of_file(self, file):
            return {
                "auth.py": ["used.py", "unused.py", "missing.py"],
                "used.py": ["entry.py"],
                "entry.py": ["auth.py"],
            }.get(file, [])

    result = symbol_impact(Store(), tmp_path, "auth.py", "validate")
    assert result["direct_ids"] == ["used.py"]
    assert result["transitive_ids"] == ["entry.py"]
    assert result["blast_score"] == 1.5
    assert result["complete"] is False


def test_source_root_and_from_module_import():
    for source in [
        "from pkg import auth as a\na.validate()\n",
        "from pkg.auth import validate\nvalidate()\n",
    ]:
        result = python_usage(source, "src/pkg/api.py", "src/pkg/auth.py", "validate")
        assert result["usage"] == "referenced"
