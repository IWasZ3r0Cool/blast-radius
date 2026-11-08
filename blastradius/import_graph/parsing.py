# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

"""Syntax extraction only. Cached results never depend on repository layout."""

import ast
import re
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Reference:
    specifier: str
    kind: str = "import"
    names: tuple[str, ...] = ()
    level: int = 0


@dataclass(frozen=True)
class Parsed:
    references: tuple[Reference, ...]
    diagnostics: tuple[str, ...] = ()


@lru_cache(maxsize=512)
def python_imports(source: bytes) -> Parsed:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, UnicodeError) as exc:
        return Parsed((), (f"Python parse error: {exc}",))
    references = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.extend(Reference(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            references.append(
                Reference(
                    node.module or "",
                    "from",
                    tuple(alias.name for alias in node.names),
                    node.level,
                )
            )
    return Parsed(tuple(references))


@lru_cache(maxsize=3)
def _language(dialect: str):
    from tree_sitter import Language

    if dialect == "js":
        import tree_sitter_javascript

        return Language(tree_sitter_javascript.language())
    import tree_sitter_typescript

    return Language(
        tree_sitter_typescript.language_tsx()
        if dialect == "tsx"
        else tree_sitter_typescript.language_typescript()
    )


@lru_cache(maxsize=512)
def javascript_imports(source: bytes, dialect: str = "js") -> Parsed:
    import json5
    from tree_sitter import Parser

    tree = Parser(_language(dialect)).parse(source)
    references = []
    diagnostics = (
        ["JavaScript/TypeScript parse error; complete imports were retained"]
        if tree.root_node.has_error
        else []
    )
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.named_children))
        literal = None
        if node.type in {"import_statement", "export_statement"}:
            literal = node.child_by_field_name("source")
        elif node.type == "call_expression":
            function = node.child_by_field_name("function")
            if function is None or function.type not in {"import", "identifier"}:
                continue
            if function.text not in {b"import", b"require"}:
                continue
            arguments = node.child_by_field_name("arguments")
            children = (
                [child for child in arguments.named_children if child.type != "comment"]
                if arguments
                else []
            )
            if children:
                literal = children[0]
                if literal.type not in {"string", "template_string"}:
                    diagnostics.append("Computed import cannot be resolved statically")
                    continue
        elif node.type in {"external_module_reference", "import_require_clause"}:
            literal = next(
                (child for child in node.named_children if child.type == "string"), None
            )
        if literal is None or node.has_error or literal.has_error:
            continue
        text = literal.text.decode("utf-8", errors="replace")
        if literal.type == "template_string":
            if any(
                child.type == "template_substitution"
                for child in literal.named_children
            ):
                diagnostics.append("Computed import cannot be resolved statically")
                continue
            text = '"' + text[1:-1].replace('"', '\\"') + '"'
        try:
            specifier = json5.loads(text)
        except ValueError:
            diagnostics.append("Invalid import string")
            continue
        if isinstance(specifier, str):
            references.append(Reference(specifier))
    return Parsed(tuple(dict.fromkeys(references)), tuple(dict.fromkeys(diagnostics)))


_SCRIPTS = re.compile(rb"<script\b([^>]*)>(.*?)</script\s*>", re.DOTALL | re.IGNORECASE)


def parse(path: str, source: bytes) -> Parsed:
    extension = path.rsplit(".", 1)[-1].lower()
    if extension == "py":
        return python_imports(source)
    if extension == "vue":
        references, diagnostics = [], []
        for match in _SCRIPTS.finditer(source):
            attrs = match.group(1)
            dialect = (
                "tsx"
                if re.search(rb"\blang\s*=\s*['\"]tsx['\"]", attrs)
                else "ts"
                if re.search(rb"\blang\s*=\s*['\"]ts['\"]", attrs)
                else "js"
            )
            result = javascript_imports(match.group(2), dialect)
            references.extend(result.references)
            diagnostics.extend(result.diagnostics)
        return Parsed(
            tuple(dict.fromkeys(references)), tuple(dict.fromkeys(diagnostics))
        )
    return javascript_imports(
        source,
        "tsx"
        if extension == "tsx"
        else "ts"
        if extension in {"ts", "mts", "cts"}
        else "js",
    )
