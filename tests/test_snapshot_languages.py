# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from blastradius.analyze import analyze
from blastradius.import_graph import ImportGraph, Snapshot
from blastradius.import_graph.go import GoGraph
from blastradius.import_graph.parsing import parse
from tests.test_language_analyzers import _write


def graph_for(sources):
    contents = {p: s.encode() for p, s in sources.items()}
    return ImportGraph(Snapshot(contents, contents.__getitem__))


def assert_parity(root, sources):
    for path, source in sources.items():
        _write(root, path, source)
    expected_nodes, expected_edges = graph_for(sources).graph()
    data = analyze(str(root))
    internal = {n["id"] for n in data["nodes"] if n["type"] != "import"}
    assert internal == expected_nodes
    assert {
        (e["source"], e["target"])
        for e in data["links"]
        if e["source"] in internal and e["target"] in internal
    } == expected_edges
    return data


def go_sources():
    return {
        "go.work": "go 1.22\nuse (\n ./app\n ./one\n)\n",
        "app/go.mod": "module example.com/app\nrequire example.com/shared v0.0.0\n",
        "app/main.go": 'package main\nimport "example.com/shared/client"\n',
        "one/go.mod": "module example.com/shared\n",
        "one/client/client.go": "package client\n",
        "two/go.mod": "module example.com/two\n",
        "two/client/client.go": "package client\n",
    }


def test_go_workspace_membership_and_independent_nonmembers(tmp_path):
    sources = go_sources()
    sources["two/main.go"] = 'package two\nimport "example.com/shared/client"\n'
    sources["one/nested/go.mod"] = "module example.com/shared/nested\n"
    sources["one/nested/lib.go"] = "package nested\n"
    sources["app/main.go"] += 'import "example.com/shared/nested"\n'
    data = assert_parity(tmp_path, sources)
    internal = graph_for(sources).graph()[1]
    assert ("app", "one/client") in internal
    assert ("two", "one/client") not in internal
    assert ("app", "one/nested") not in internal
    assert {"app", "two", "one/nested"} <= {n["id"] for n in data["nodes"]}
    sources["go.work"] += "use ./one/nested\n"
    assert ("app", "one/nested") in graph_for(sources).graph()[1]


def test_go_nearest_workspace_and_member_replacement_conflicts(tmp_path):
    sources = go_sources()
    sources["one/go.mod"] += "replace example.com/upstream => ../one\n"
    sources["app/go.mod"] += "replace example.com/upstream => ../two\n"
    sources["app/main.go"] = 'package main\nimport "example.com/upstream/client"\n'
    graph = graph_for(sources)
    assert graph.graph()[1] == set()
    assert any("conflicting local replacements" in d for d in graph.diagnostics)
    sources["go.work"] += "replace example.com/upstream => ./one\n"
    assert ("app", "one/client") in assert_parity_edges(tmp_path, sources)
    # A nearer workspace with no listed member makes app standalone.
    sources["app/go.work"] = (
        "go 1.22\nuse ../two\nreplace example.com/upstream => ../one\n"
    )
    edges = graph_for(sources).graph()[1]
    assert ("app", "two/client") in edges
    assert ("app", "one/client") not in edges


def assert_parity_edges(root, sources):
    assert_parity(root, sources)
    return graph_for(sources).graph()[1]


def test_go_package_metadata_counts_and_edge_weights(tmp_path):
    sources = {
        "go.mod": "module example.com/app\nrequire example.com/dependency v1.0.0\n",
        "main.go": 'package main\nimport "example.com/app/lib"\nimport "example.com/dependency/a"\n',
        "extra.go": 'package main\nimport "example.com/app/lib"\nimport "example.com/dependency/b"\n',
        "lib/value.go": "package lib\n",
    }
    data = assert_parity(tmp_path, sources)
    root = next(n for n in data["nodes"] if n["id"] == ".")
    assert root["imports"] == 3
    assert root["loc"] == 8
    weights = {(e["source"], e["target"]): e["weight"] for e in data["links"]}
    assert weights[(".", "lib")] == 1
    assert weights[(".", "example.com/dependency")] == 2
    assert not any(n["id"].endswith(".go") for n in data["nodes"])


def test_go_local_replacement_directory_preserves_static_resolution(tmp_path):
    sources = {
        "go.mod": "module example.com/app\nreplace example.com/shared => ./local\n",
        "main.go": 'package main\nimport "example.com/shared/client"\n',
        "local/client/client.go": "package client\n",
    }
    assert (".", "local/client") in assert_parity_edges(tmp_path, sources)
    # A nested module owns its packages even when its parent directory is the
    # replacement target; it must not be claimed through that parent.
    sources["local/client/go.mod"] = "module example.com/different\n"
    assert (".", "local/client") not in graph_for(sources).graph()[1]


@pytest.mark.parametrize("replacement", ["/outside", "../../outside", "C:/outside"])
def test_outside_paths_never_resolve_from_host(tmp_path, replacement):
    sources = go_sources()
    sources["go.work"] += f"replace example.com/shared => {replacement}\n"
    sources.update(
        {
            "Cargo.toml": f'[package]\nname="demo"\nversion="0.1.0"\n[dependencies]\nother={{path="{replacement}"}}\n',
            "src/lib.rs": f'#[path="{replacement}/value.rs"] mod value;\nuse other::Item;\n',
        }
    )
    graph = graph_for(sources)
    _, edges = graph.graph()
    assert edges == set()
    assert any("unsupported repository path" in d for d in graph.diagnostics)
    assert_parity(tmp_path, sources)


def test_rust_targets_variants_and_cargo_alias_parity(tmp_path):
    sources = {
        "Cargo.toml": '[workspace]\nmembers=["app","domain"]\n[workspace.dependencies]\nmodel={path="domain",package="domain-core"}\n',
        "domain/Cargo.toml": '[package]\nname="domain-core"\nversion="0.1.0"\n[lib]\npath="code/root.rs"\n',
        "domain/code/root.rs": "pub mod models;\n",
        "domain/code/models.rs": "pub struct Item;\n",
        "app/Cargo.toml": '[package]\nname="app"\nversion="0.1.0"\n[dependencies]\nmodel.workspace=true\n',
        "app/src/lib.rs": '#[cfg(unix)]\n#[path="unix.rs"] mod platform;\n#[cfg(windows)]\n#[path="windows.rs"] mod platform;\nmod consumer;\n',
        "app/src/unix.rs": "pub mod detail;\n",
        "app/src/windows.rs": "pub mod detail;\n",
        "app/src/unix/detail.rs": "pub struct Item;\n",
        "app/src/windows/detail.rs": "pub struct Item;\n",
        "app/src/consumer.rs": "pub use crate::platform::detail::Item;\nuse model::models::Item as Model;\n",
        "app/examples/inspect/main.rs": "use app::platform::detail::Item;\n",
    }
    data = assert_parity(tmp_path, sources)
    consumer = {
        e["target"] for e in data["links"] if e["source"] == "app/src/consumer.rs"
    }
    assert consumer == {
        "app/src/unix/detail.rs",
        "app/src/windows/detail.rs",
        "domain/code/models.rs",
    }
    assert all(e["weight"] == 1 for e in data["links"])


def test_rust_module_cycles_terminate_without_losing_healthy_modules():
    graph = graph_for(
        {
            "src/lib.rs": '#[path="lib.rs"] mod recursive;\nmod value;\n',
            "src/value.rs": "pub struct Item;\n",
        }
    )
    assert ("src/lib.rs", "src/value.rs") in graph.graph()[1]
    assert any("cyclic Rust module" in d for d in graph.diagnostics)


@pytest.mark.parametrize(
    ("filename", "source"), [("lib.go", b"package lib\n"), ("lib.rs", b"mod value;\n")]
)
def test_language_parsed_records_are_immutable(filename, source):
    parsed = parse(filename, source)
    with pytest.raises(FrozenInstanceError):
        parsed.loc = 100
    assert isinstance(parse(filename, source), type(parsed))


def test_shared_analysis_does_not_import_analyzers_or_semantic():
    root = Path(__file__).resolve().parents[1] / "blastradius"
    for directory in ("import_graph", "temporal", "store"):
        for file in (root / directory).glob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                names = (
                    [n.name for n in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                assert not any(
                    name.startswith(("blastradius.analyzers", "blastradius.semantic"))
                    for name in names
                ), file


def test_ignored_manifests_and_sources_are_not_resolution_candidates():
    sources = go_sources()
    sources[".gitignore"] = "one/\n"
    graph = graph_for(sources)
    assert graph.graph()[1] == set()
    assert "one/client" not in graph.graph()[0]
    results = GoGraph(graph.snapshot).results
    assert results["app"].imports.external == ("example.com/shared",)


def test_installed_package_supports_cargo_without_optional_extra(
    installed_package, tmp_path
):
    from tests.package_support import run

    _write(tmp_path, "Cargo.toml", '[package]\nname="demo"\nversion="0.1.0"\n')
    _write(tmp_path, "src/lib.rs", "mod value;\n")
    _write(tmp_path, "src/value.rs", "pub struct Item;\n")
    output = run(
        [
            str(installed_package.python),
            "-c",
            (
                "import sys; from blastradius.analyze import analyze; "
                'g=analyze(sys.argv[1]); assert any(e["source"]=="src/lib.rs" and e["target"]=="src/value.rs" for e in g["links"]); print("ok")'
            ),
            str(tmp_path),
        ],
        tmp_path,
        env=installed_package.env,
    )
    assert output.strip() == "ok"
