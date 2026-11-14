# Copyright 2026 Christopher Wilson
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from pathlib import Path

import pytest

from blastradius.analyze import detect_languages
from blastradius.analyzers.base import is_skip_dir
from blastradius.import_graph.snapshot import skipped


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_empty_tree_has_no_languages(tmp_path: Path) -> None:
    assert detect_languages(tmp_path) == []


@pytest.mark.parametrize(
    "directory",
    [
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".git",
        "node_modules",
        "dist",
        "build",
        ".next",
        "target",
        "vendor",
        ".bundle",
    ],
)
def test_nested_excluded_directories_do_not_trigger_detection(
    tmp_path: Path, directory: str
) -> None:
    _touch(tmp_path, f"nested/{directory}/deeper/module.py")
    assert detect_languages(tmp_path) == []


@pytest.mark.parametrize(
    ("source", "excluded"),
    [
        ("out/module.py", True),
        ("coverage/module.py", True),
        (".generated/module.py", True),
        ("src/.example.py", True),
        ("src/vendor/module.py", True),
        ("build_tools/module.py", False),
        ("src/Out/module.py", False),
    ],
)
def test_detection_and_collection_agree_on_path_exclusions(
    tmp_path: Path, source: str, excluded: bool
) -> None:
    _touch(tmp_path, source)
    assert is_skip_dir(Path(source)) == excluded
    assert skipped(source) == excluded
    assert detect_languages(tmp_path) == ([] if excluded else ["python"])


def test_detection_does_not_apply_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("generated/\n", encoding="utf-8")
    _touch(tmp_path, "generated/module.py")
    assert detect_languages(tmp_path) == ["python"]


@pytest.mark.parametrize("extension", ["js", "ts"])
def test_excluded_matches_do_not_prevent_eligible_matches(
    tmp_path: Path, extension: str
) -> None:
    _touch(tmp_path, "coverage/report.js")
    assert detect_languages(tmp_path) == []
    _touch(tmp_path, f"src/application.{extension}")
    assert detect_languages(tmp_path) == ["javascript"]


def test_multiple_languages_retain_dispatch_order(tmp_path: Path) -> None:
    for source in [
        "nested/main.tf",
        "nested/schema.prisma",
        "nested/Dockerfile.dev",
        "nested/app.php",
        "nested/App.kt",
        "nested/lib.rs",
        "nested/app.rb",
        "nested/main.go",
        "nested/style.scss",
        "nested/App.vue",
        "nested/app.py",
    ]:
        _touch(tmp_path, source)
    _touch(tmp_path, ".github/workflows/check.yml")
    assert detect_languages(tmp_path) == [
        "python",
        "javascript",
        "css",
        "go",
        "ruby",
        "rust",
        "java",
        "php",
        "docker",
        "ci",
        "schema",
        "terraform",
    ]


@pytest.mark.parametrize(
    ("marker", "language"),
    [
        ("package.json", "javascript"),
        ("go.mod", "go"),
        ("Gemfile", "ruby"),
        ("Cargo.toml", "rust"),
        ("composer.json", "php"),
        ("docker-compose.yml", "docker"),
        ("docker-compose.yaml", "docker"),
        ("compose.yml", "docker"),
        ("compose.yaml", "docker"),
        (".github/workflows/", "ci"),
        (".gitlab-ci.yml", "ci"),
        (".gitlab-ci.yaml", "ci"),
    ],
)
def test_root_markers_trigger_detection_without_source(
    tmp_path: Path, marker: str, language: str
) -> None:
    if marker.endswith("/"):
        (tmp_path / marker).mkdir(parents=True)
    else:
        _touch(tmp_path, marker)
    assert detect_languages(tmp_path) == [language]


def test_exclusion_checks_use_repository_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "build" / "repo"
    _touch(root, "src/module.py")
    _touch(root, "build/ignored.js")
    assert detect_languages(root) == ["python"]
    # Manifest checks remain independent of the source-path exclusions.
    _touch(root, "package.json")
    assert detect_languages(root) == ["python", "javascript"]


def test_glob_matching_still_accepts_directory_names(tmp_path: Path) -> None:
    (tmp_path / "module.py").mkdir()
    assert detect_languages(tmp_path) == ["python"]
