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
from email.parser import Parser
from http.client import HTTPConnection
from pathlib import Path
from typing import NamedTuple

import pytest

ROOT = Path(__file__).resolve().parents[1]


class InstalledTool(NamedTuple):
    python: Path
    executable: Path
    env: dict[str, str]
    sdist: Path
    wheel: Path


def _environment() -> dict[str, str]:
    # Do not let an editable checkout leak into the installed-package test.
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
        and not key.startswith("BLASTRADIUS_EMBEDDING_")
    }


def _run(
    command: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env if env is not None else _environment(),
        input=stdin,
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
) -> InstalledTool:
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
    tool_dir = root / "tools"
    bin_dir = root / "bin"
    env = {
        **_environment(),
        "UV_TOOL_DIR": str(tool_dir),
        "UV_TOOL_BIN_DIR": str(bin_dir),
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
    }
    _run(
        [uv, "tool", "install", "--no-index", "--python", sys.executable, str(wheel)],
        root,
        env=env,
    )
    assert Path(_run([uv, "tool", "dir", "--bin"], root, env=env).strip()) == bin_dir
    environment = tool_dir / "blastradius-cli"
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    executable = bin_dir / ("blastradius.exe" if os.name == "nt" else "blastradius")
    assert shutil.which("blastradius", path=env["PATH"]) == str(executable)
    return InstalledTool(python, executable, env, sdist, wheel)


def _get(port: int, path: str) -> tuple[int, str, bytes]:
    connection = HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type", ""), response.read()
    finally:
        connection.close()


@contextmanager
def _running_cli(
    command: list[str],
    root: Path,
    *,
    source: bool = False,
    env: dict[str, str] | None = None,
    extra_args: tuple[str, ...] = (),
):
    repo = root / "repo"
    shutil.copytree(ROOT / "tests/fixtures/simple_python", repo)
    # Reserve an OS-assigned port rather than assuming 8080 is available.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = dict(env) if env is not None else _environment()
    if source:
        env["PYTHONPATH"] = str(ROOT)
    log_path = root / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                *command,
                "serve",
                "--viz",
                "--repo",
                str(repo),
                "--port",
                str(port),
                *extra_args,
            ],
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
    installed_package: InstalledTool, tmp_path: Path
) -> None:
    python, cli, env, sdist, wheel = installed_package
    imported_from = Path(
        _run(
            [
                str(python),
                "-I",
                "-c",
                "import blastradius; print(blastradius.__file__)",
            ],
            tmp_path,
            env=env,
        ).strip()
    ).resolve()
    assert python.parent.parent.resolve() in imported_from.parents
    with _running_cli([str(cli)], tmp_path, env=env) as (port, repo):
        _assert_routes(port, repo)

    with tarfile.open(sdist) as archive:
        assert any(
            name.endswith("/blastradius/static/explorer.html")
            for name in archive.getnames()
        )
    with zipfile.ZipFile(wheel) as archive:
        assert "blastradius/static/explorer.html" in archive.namelist()
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
        assert set(metadata.get_all("Project-URL")) == {
            "Homepage, https://github.com/IWasZ3r0Cool/blast-radius",
            "Repository, https://github.com/IWasZ3r0Cool/blast-radius",
            "Issues, https://github.com/IWasZ3r0Cool/blast-radius/issues",
        }


def test_uv_tool_quickstart_commands(
    installed_package: InstalledTool, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "tests/fixtures/simple_python", repo)
    cli = str(installed_package.executable)
    env = installed_package.env
    assert "serve" in _run([cli, "--help"], repo, env=env)
    _run([cli, "analyze", "."], repo, env=env)
    _run([cli, "symbols", "."], repo, env=env)
    impact = json.loads(_run([cli, "impact", "models.py", "--json"], repo, env=env))
    assert impact["direct_dependents"] == 2
    lookup = json.loads(_run([cli, "lookup", "User", "--json"], repo, env=env))
    assert lookup["matches"][0]["file"] == "models.py"
    search = json.loads(_run([cli, "search", "User", "--json"], repo, env=env))
    assert any(symbol["name"] == "User" for symbol in search["symbols"])
    _run([cli, "impact", "models.py", "--out", "impact.md"], repo, env=env)
    report = (repo / "impact.md").read_text(encoding="utf-8")
    assert "https://github.com/IWasZ3r0Cool/blast-radius" in report


def test_uv_tool_runs_documented_mcp_command(
    installed_package: InstalledTool, tmp_path: Path
) -> None:
    request = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
    response = json.loads(
        _run(
            [str(installed_package.executable), "serve", "--mcp"],
            tmp_path,
            env=installed_package.env,
            stdin=request,
        )
    )
    assert response["id"] == 1
    assert {"analyze_repo", "get_impact", "lookup_symbol"} <= {
        tool["name"] for tool in response["result"]["tools"]
    }


def test_uvx_explicit_package_and_command(
    installed_package: InstalledTool, tmp_path: Path
) -> None:
    uvx = shutil.which("uvx")
    assert uvx is not None, "uv provides the uvx command"
    # Use the locally built wheel instead of downloading a published release.
    output = _run(
        [
            uvx,
            "--isolated",
            "--no-index",
            "--python",
            sys.executable,
            "--from",
            str(installed_package.wheel),
            "blastradius",
            "--help",
        ],
        tmp_path,
        env=installed_package.env,
    )
    assert "usage: blastradius" in output
    assert "serve" in output


def test_tool_watch_install_hint(
    installed_package: InstalledTool, tmp_path: Path
) -> None:
    result = subprocess.run(
        [str(installed_package.executable), "analyze", ".", "--watch"],
        cwd=tmp_path,
        env=installed_package.env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert "uv tool install --force 'blastradius-cli[watch]'" in result.stderr
    assert "pip install" not in result.stderr


def test_tool_visualizer_watch_install_hint(
    installed_package: InstalledTool, tmp_path: Path
) -> None:
    with _running_cli(
        [str(installed_package.executable)],
        tmp_path,
        env=installed_package.env,
        extra_args=("--watch",),
    ) as (port, _):
        assert _get(port, "/")[0] == 200
    log = (tmp_path / "server.log").read_text(encoding="utf-8")
    assert "uv tool install --force 'blastradius-cli[watch]'" in log
    assert "pip install" not in log


def test_tool_semantic_install_hint(
    installed_package: InstalledTool, tmp_path: Path
) -> None:
    result = subprocess.run(
        [
            str(installed_package.python),
            "-I",
            "-c",
            (
                "import sys; from pathlib import Path; from blastradius.store import Store; "
                "store = Store(Path(sys.argv[1])); print(store.init_vectors(3)); store.close()"
            ),
            str(tmp_path / "index.db"),
        ],
        cwd=tmp_path,
        env=installed_package.env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "False"
    assert "uv tool install --force 'blastradius-cli[semantic]'" in result.stderr
    assert "pip install" not in result.stderr


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
