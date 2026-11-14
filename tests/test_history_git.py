# Copyright 2026 Christopher Wilson
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

import subprocess
from unittest.mock import Mock, call

import pytest

from blastradius.temporal import history


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["git", "cat-file", "--batch"], 60),
        KeyboardInterrupt(),
        SystemExit(7),
    ],
    ids=["timeout", "keyboard-interrupt", "system-exit"],
)
def test_blob_reader_reaps_child_and_reraises_original_error(
    tmp_path, monkeypatch, error
):
    process = Mock(spec=subprocess.Popen)
    process.communicate.side_effect = [error, (b"", b"")]
    monkeypatch.setattr(history.subprocess, "Popen", Mock(return_value=process))

    with pytest.raises(type(error)) as raised:
        history._git_cat_file_batch(tmp_path, ["abc"])

    assert raised.value is error
    assert process.mock_calls == [
        call.communicate(b"abc\n", timeout=60),
        call.kill(),
        call.communicate(),
    ]
