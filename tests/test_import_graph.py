# Copyright 2026 Christopher Wilson
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from pathlib import Path

import pytest

from blastradius.analyze import analyze
from blastradius.import_graph import ImportGraph, Snapshot
from blastradius.import_graph.parsing import javascript_imports, parse, python_imports


def graph_for(sources):
    content = {
        path: value.encode() if isinstance(value, str) else value
        for path, value in sources.items()
    }
    return ImportGraph(Snapshot(content, content.__getitem__))


@pytest.mark.parametrize(
    ("path", "source", "files", "expected"),
    [
        (
            "pkg/main.py",
            "from . import utils",
            ["pkg/__init__.py", "pkg/utils.py"],
            {"pkg/__init__.py", "pkg/utils.py"},
        ),
        (
            "main.py",
            "from pkg import utils",
            ["pkg/__init__.py", "pkg/utils.py"],
            {"pkg/__init__.py", "pkg/utils.py"},
        ),
        (
            "src/pkg/main.py",
            "from pkg.utils import VALUE",
            ["src/pkg/utils.py"],
            {"src/pkg/utils.py"},
        ),
        (
            "pkg/sub/main.py",
            "from ..utils import VALUE",
            ["pkg/utils.py"],
            {"pkg/utils.py"},
        ),
        ("pkg/main.py", "from ...utils import VALUE", ["utils.py"], set()),
        ("main.py", "from pkg import VALUE", ["pkg/__init__.py"], {"pkg/__init__.py"}),
        ("pkg/main.py", "from . import utils", ["pkg/utils.py"], {"pkg/utils.py"}),
        (
            "main.py",
            "import pkg.utils",
            ["pkg/__init__.py", "pkg/utils.py"],
            {"pkg/__init__.py", "pkg/utils.py"},
        ),
    ],
)
def test_python_module_resolution(path, source, files, expected):
    graph = graph_for({**dict.fromkeys(files, "VALUE = 1"), path: source})
    assert set(graph.analyze(path).targets) == expected


def test_python_encoding_and_parse_diagnostics():
    graph = graph_for(
        {"main.py": b"# coding: cp1252\n# caf\xe9\nimport utils\n", "utils.py": ""}
    )
    assert graph.analyze("main.py").targets == ("utils.py",)
    graph = graph_for({"main.py": "import (", "utils.py": ""})
    assert "Python parse error" in " ".join(graph.analyze("main.py").diagnostics)


def test_normal_index_retains_malformed_python_file_and_healthy_siblings(
    tmp_path, capsys
):
    (tmp_path / "broken.py").write_text("import (", encoding="utf-8")
    (tmp_path / "healthy.py").write_text("from target import VALUE\n", encoding="utf-8")
    (tmp_path / "target.py").write_text("VALUE = 1\n", encoding="utf-8")

    data = analyze(str(tmp_path))

    assert {node["id"] for node in data["nodes"]} == {
        "broken.py",
        "healthy.py",
        "target.py",
    }
    assert {(link["source"], link["target"]) for link in data["links"]} == {
        ("healthy.py", "target.py")
    }
    output = capsys.readouterr()
    assert "broken.py: Python parse error" in output.err
    assert "analyzer failed" not in output.err
    assert "Traceback" not in output.err
    assert output.out == ""


def test_unresolved_absolute_python_package_is_local_and_diagnostic_is_repeatable():
    graph = graph_for({"main.py": "import pkg.missing", "pkg/__init__.py": ""})
    result = graph.analyze("main.py")
    assert not result.targets
    assert not result.external
    assert "unresolved local import pkg.missing" in " ".join(result.diagnostics)
    assert graph.analyze("main.py").diagnostics == result.diagnostics


def test_relative_import_cannot_escape_src_package():
    graph = graph_for(
        {"src/pkg/main.py": "from ..utils import VALUE", "src/utils.py": ""}
    )
    assert not graph.analyze("src/pkg/main.py").targets


def test_symbol_import_does_not_treat_a_module_as_a_package():
    graph = graph_for(
        {
            "main.py": "from module import VALUE",
            "module.py": "VALUE = 1",
            "module/VALUE.py": "",
        }
    )
    assert graph.analyze("main.py").targets == ("module.py",)


def test_dotted_js_module_stems_and_python_files():
    graph = graph_for(
        {
            "main.ts": 'import "./util.shared"; import "./util.py";',
            "util.shared.ts": "",
            "util.py": "",
        }
    )
    assert graph.analyze("main.ts").targets == ("util.shared.ts",)


def test_js_syntax_extracts_real_imports_only():
    source = """
// import './comment';
/* export * from './comment'; */
const text = "from './comment'";
const regex = /import 'comment'/;
import {
  value,
} from './static';
export * as ns from './reexport';
const lazy = import(/* webpackChunkName: "chunk" */ './dynamic');
const common = require(/* comment */ './common');
const template = import(`./template`);
import type { User } from './types.js';
import legacy = require('./legacy');
const runtime = import(`./${name}`);
const computed = require(prefix + name);
const view = <div title="import './comment'" />;
"""
    names = {
        name: ""
        for name in [
            "static",
            "reexport",
            "dynamic",
            "common",
            "template",
            "types",
            "legacy",
            "comment",
        ]
    }
    graph = graph_for(
        {
            "src/main.tsx": source,
            **{f"src/{name}.ts": value for name, value in names.items()},
        }
    )
    result = graph.analyze("src/main.tsx")
    assert set(result.targets) == {
        f"src/{name}.ts" for name in names if name != "comment"
    }
    assert result.count == 7
    assert any("Computed import" in message for message in result.diagnostics)
    assert not result.external


def test_relative_paths_extensions_indexes_and_vue_scripts():
    graph = graph_for(
        {
            "src/main.mts": 'import "../lib/util.js"; import "./widget"; import "../../escape";',
            "lib/util.ts": "",
            "src/widget/index.tsx": "",
            "src/main.vue": '<template>import "./ghost"</template><script>import "./one";</script><script setup lang="ts">import type { X } from "./two";</script>',
            "src/one.js": "",
            "src/two.ts": "",
            "src/ghost.js": "",
            "src/client.cts": 'import x = require("./native.cjs");',
            "src/native.cts": "",
        }
    )
    assert set(graph.analyze("src/main.mts").targets) == {
        "lib/util.ts",
        "src/widget/index.tsx",
    }
    assert set(graph.analyze("src/main.vue").targets) == {"src/one.js", "src/two.ts"}
    assert graph.analyze("src/client.cts").targets == ("src/native.cts",)


def test_aliases_use_nearest_configuration_and_local_extends():
    graph = graph_for(
        {
            "tsconfig.json": '{// comment\n"compilerOptions":{"baseUrl":".", "paths":{"@/*":["missing/*","src/*"], "exact":["src/target.ts"],},},}',
            "src/main.ts": 'import "@/target"; import "exact"; import "lib/base";',
            "src/target.ts": "",
            "lib/base.ts": "",
            "apps/web/tsconfig.json": '{"extends":"../../tsconfig", "compilerOptions":{"paths":{"@/*":["local/*"]}}}',
            "apps/web/main.ts": 'import "@/target";',
            "local/target.ts": "",
        }
    )
    assert graph.analyze("src/main.ts").targets == (
        "src/target.ts",
        "src/target.ts",
        "lib/base.ts",
    )
    assert graph.analyze("apps/web/main.ts").targets == ("local/target.ts",)
    assert not graph.diagnostics


def test_package_exports_private_imports_and_all_local_variants():
    graph = graph_for(
        {
            "package.json": '{"name":"app", "imports":{"#private":"./src/private.ts"}}',
            "src/main.ts": 'import "@acme/lib"; import "@acme/lib/features/one"; import "@acme/lib/hidden"; import "#private"; import "react";',
            "packages/lib/package.json": '{"name":"@acme/lib", "exports":{".":{"browser":"./browser.ts", "node":"./node.ts", "types":"./types.d.ts"}, "./features/*":"./src/*.ts"}}',
            "packages/lib/browser.ts": "",
            "packages/lib/node.ts": "",
            "packages/lib/types.d.ts": "",
            "packages/lib/src/one.ts": "",
            "packages/lib/hidden.ts": "",
            "src/private.ts": "",
        }
    )
    result = graph.analyze("src/main.ts")
    assert set(result.targets) == {
        "packages/lib/browser.ts",
        "packages/lib/node.ts",
        "packages/lib/types.d.ts",
        "packages/lib/src/one.ts",
        "src/private.ts",
    }
    assert result.external == ("react",)
    assert any("@acme/lib/hidden" in message for message in result.diagnostics)


def test_directory_package_main_and_legacy_package_fields():
    graph = graph_for(
        {
            "main.ts": 'import "./directory"; import "lib";',
            "directory/package.json": '{"main":"entry.js"}',
            "directory/entry.ts": "",
            "lib/package.json": '{"name":"lib", "main":"main.js", "types":"index.d.ts"}',
            "lib/main.ts": "",
            "lib/index.d.ts": "",
        }
    )
    assert set(graph.analyze("main.ts").targets) == {
        "directory/entry.ts",
        "lib/main.ts",
        "lib/index.d.ts",
    }


def test_bad_or_cyclic_configuration_does_not_invent_edges():
    graph = graph_for(
        {
            "main.ts": 'import "@/target"; import "lib";',
            "tsconfig.json": '{"extends":"./base.json"}',
            "base.json": '{"extends":"./tsconfig.json"}',
            "one/package.json": '{"name":"lib"}',
            "two/package.json": '{"name":"lib"}',
            "invalid/package.json": "[1]",
            "target.ts": "",
        }
    )
    assert not graph.analyze("main.ts").targets
    assert any("cyclic configuration" in message for message in graph.diagnostics)
    assert any("ambiguous local package" in message for message in graph.diagnostics)
    assert any("invalid configuration" in message for message in graph.diagnostics)


def test_error_recovery_retains_complete_imports():
    graph = graph_for({"main.ts": 'import "./good"; const broken = ;', "good.ts": ""})
    result = graph.analyze("main.ts")
    assert result.targets == ("good.ts",)
    assert any("parse error" in message for message in result.diagnostics)


def test_cached_parsing_is_independent_of_resolution_context():
    javascript_imports.cache_clear()
    source = 'import "@/target";'
    for target in ["one", "two"]:
        graph = graph_for(
            {
                "main.ts": source,
                "tsconfig.json": '{"compilerOptions":{"paths":{"@/*":["'
                + target
                + '/*"]}}}',
                f"{target}/target.ts": "",
            }
        )
        assert graph.analyze("main.ts").targets == (f"{target}/target.ts",)
    assert javascript_imports.cache_info().misses == 1
    assert javascript_imports.cache_info().hits == 1
    python_imports.cache_clear()
    graph_for({"a.py": "import b", "b.py": ""}).analyze("a.py")
    graph_for({"a.py": "import b", "src/b.py": ""}).analyze("a.py")
    assert python_imports.cache_info().hits == 1


@pytest.mark.parametrize("source", [b"import b", b"import b\nbroken ="])
def test_preparsed_sources_reuse_syntax_with_fallback_and_explicit_source(source):
    sources = {"a.py": source, "b.py": b"", "c.py": b""}
    expected = graph_for(sources).analyze("a.py")
    parsed = parse("a.py", source)

    def read(path):
        assert path != "a.py", "a pre-parsed source should not be read again"
        return sources[path]

    graph = ImportGraph(Snapshot(sources, read), parsed_sources={"a.py": parsed})
    assert graph.analyze("a.py") == expected
    assert graph.analyze("b.py").targets == ()
    assert graph.analyze("a.py", b"import c").targets == ("c.py",)


def test_normal_index_and_snapshot_share_internal_edges(tmp_path: Path):
    sources = {
        "src/pkg/main.py": "from . import utils",
        "src/pkg/utils.py": "",
        "src/pkg/__init__.py": "",
        "src/main.ts": 'export * as lib from "../lib/index.js"; import "./style.css";',
        "lib/index.ts": "",
        "src/style.css": "body {}",
        "ignored.py": "import missing",
        ".gitignore": "ignored.py\n",
        "node_modules/ghost.ts": 'import "fake";',
    }
    for path, source in sources.items():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")
    normal = analyze(str(tmp_path))
    ids = {node["id"] for node in normal["nodes"] if node["type"] != "import"}
    normal_edges = {
        (edge["source"], edge["target"])
        for edge in normal["links"]
        if edge["source"] in ids and edge["target"] in ids
    }
    nodes, edges = graph_for(sources).graph()
    assert ids == nodes
    assert normal_edges == edges
