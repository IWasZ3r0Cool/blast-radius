# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import zipfile
from contextlib import contextmanager
from http.client import HTTPConnection
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict[str, str]:
    # Do not let an editable checkout leak into the installed-package test.
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }


def _run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def installed_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    uv = shutil.which("uv")
    assert uv is not None, "The packaging integration test requires uv"
    root = tmp_path_factory.mktemp("viz-package")
    source = root / "source"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, source / name)
    shutil.copytree(
        ROOT / "blastradius",
        source / "blastradius",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    artifacts = root / "dist"
    _run([uv, "build", "--sdist", str(source), "--out-dir", str(artifacts)], root)
    sdist = next(artifacts.glob("*.tar.gz"))
    _run([uv, "build", "--wheel", str(sdist), "--out-dir", str(artifacts)], root)
    wheel = next(artifacts.glob("*.whl"))
    environment = root / "venv"
    _run(
        [uv, "venv", "--no-project", "--python", sys.executable, str(environment)], root
    )
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    _run([uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)], root)
    return python, sdist, wheel


def _get(port: int, path: str) -> tuple[int, str, bytes]:
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type", ""), response.read()
    finally:
        connection.close()


@contextmanager
def _running_cli(command: list[str], root: Path, *, source: bool = False):
    repo = root / "repo"
    shutil.copytree(ROOT / "tests/fixtures/simple_python", repo)
    # Reserve an OS-assigned port rather than assuming 8080 is available.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = _environment()
    if source:
        env["PYTHONPATH"] = str(ROOT)
    log_path = root / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [*command, "serve", "--viz", "--repo", str(repo), "--port", str(port)],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 10
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    _get(port, "/")
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                pytest.fail("Visualizer failed to start:\n" + log_path.read_text())
            yield port, repo
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _assert_routes(port: int, repo: Path) -> None:
    for path in ("/", "/index.html", "/?view=graph", "/index.html?view=graph"):
        status, content_type, body = _get(port, path)
        assert status == 200, f"Visualizer {path} returned {status}"
        assert content_type == "text/html; charset=utf-8"
        assert b"<title>blastradius</title>" in body

    status, content_type, body = _get(port, "/graph")
    assert status == 200
    assert content_type == "application/json"
    graph = json.loads(body)
    assert {node["id"] for node in graph["nodes"]} == {
        "main.py",
        "models.py",
        "utils.py",
    }
    assert {(link["source"], link["target"]) for link in graph["links"]} == {
        ("main.py", "models.py"),
        ("main.py", "utils.py"),
        ("utils.py", "models.py"),
    }
    assert _get(port, "/nonexistent")[0] == 404

    (repo / "blastradius.json").unlink()
    status, content_type, body = _get(port, "/graph")
    assert status == 404
    assert content_type == "application/json"
    assert "blastradius.json not found" in json.loads(body)["error"]
    assert _get(port, "/")[0] == 200


def test_source_checkout_serves_visualizer_outside_checkout(tmp_path: Path) -> None:
    with _running_cli(
        [sys.executable, "-m", "blastradius.cli"], tmp_path, source=True
    ) as (port, repo):
        _assert_routes(port, repo)


def test_installed_wheel_serves_visualizer(
    installed_package: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    python, sdist, wheel = installed_package
    imported_from = Path(
        _run(
            [
                str(python),
                "-I",
                "-c",
                "import blastradius; print(blastradius.__file__)",
            ],
            tmp_path,
        ).strip()
    ).resolve()
    assert python.parent.parent.resolve() in imported_from.parents
    cli = python.parent / ("blastradius.exe" if os.name == "nt" else "blastradius")

    with _running_cli([str(cli)], tmp_path) as (port, repo):
        _assert_routes(port, repo)

    with tarfile.open(sdist) as archive:
        assert any(
            name.endswith("/blastradius/static/explorer.html")
            for name in archive.getnames()
        )
    with zipfile.ZipFile(wheel) as archive:
        assert "blastradius/static/explorer.html" in archive.namelist()


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_unavailable_html_fails_before_startup(
    error: type[OSError],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    from blastradius.viz_server import serve

    read_bytes = Path.read_bytes

    def unavailable_html(path: Path) -> bytes:
        if path.name == "explorer.html":
            raise error("UI resource unavailable")
        return read_bytes(path)

    # Simulate filesystem failures without altering the checkout or relying on
    # platform-specific file permission enforcement (e.g. when running as root).
    monkeypatch.setattr(Path, "read_bytes", unavailable_html)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        with pytest.raises(SystemExit) as exited:
            serve(str(tmp_path), port=reservation.getsockname()[1], watch=True)

    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert "reinstall blastradius-cli" in captured.err.lower()
    assert "UI resource unavailable" in captured.err
    assert "Analyzing" not in captured.err
    assert "watch" not in captured.err
    assert "Serving at" not in captured.out
    assert not (tmp_path / "blastradius.json").exists()
    assert not (tmp_path / ".blastradius").exists()
