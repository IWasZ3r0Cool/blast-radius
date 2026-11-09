# Copyright 2026 David Scheiderman
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from blastradius.hook import install

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="Generated hook requires Bash")
@pytest.mark.parametrize(
    "strict, threshold, warned, exit_code",
    [(False, 2, True, 0), (True, 2, True, 1), (True, 3, False, 0)],
)
def test_generated_hook_passes_threshold_and_preserves_exit_behavior(
    tmp_path, strict, threshold, warned, exit_code
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "core.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "core.py"], cwd=repo, check=True, capture_output=True)
    (repo / "blastradius.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": path}
                    for path in ("core.py", "client.py", "tests/test_core.py")
                ],
                "links": [
                    {"source": path, "target": "core.py"}
                    for path in ("client.py", "tests/test_core.py")
                ],
            }
        ),
        encoding="utf-8",
    )
    install(str(repo), threshold=threshold, strict=strict)

    commands = tmp_path / "bin"
    commands.mkdir()
    invocation = tmp_path / "args.json"
    launcher = commands / "blastradius"
    launcher.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        f"Path({str(invocation)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "from blastradius.cli import main\nmain()\n"
    )
    launcher.chmod(0o755)
    # Isolate the existing GNU grep -P dependency on hosts with BSD grep.
    grep = commands / "grep"
    grep.write_text(
        f"#!{sys.executable}\nimport re, sys\n"
        "assert sys.argv[1:] == ['-oP', r'Blast Score: \\K[0-9.]+']\n"
        "match = re.search(r'Blast Score: ([0-9.]+)', sys.stdin.read())\n"
        "if match: print(match[1])\n"
    )
    grep.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(ROOT),
    }
    result = subprocess.run(
        ["bash", str(repo / ".git/hooks/pre-commit")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    assert ("HIGH BLAST RADIUS" in result.stdout) is warned
    assert ("Mitigation suggestions" in result.stdout) is warned
    assert (
        "downstream test candidates (1): `tests/test_core.py`" in result.stdout
    ) is warned
    assert ("Commit blocked" in result.stdout) is (warned and strict)
    assert json.loads(invocation.read_text()) == [
        "impact",
        "core.py",
        "--index",
        str(repo / "blastradius.json"),
        "--guidance-threshold",
        str(threshold),
    ]
