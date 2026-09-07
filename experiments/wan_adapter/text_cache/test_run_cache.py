# SPDX-License-Identifier: Apache-2.0
"""CLI contract checks with a fake subprocess; these do not encode text."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from . import run_cache


@pytest.mark.parametrize("custom", [False, True])
def test_prompt_path_forwarding_preserves_worker_default(tmp_path, monkeypatch, custom):
    output = tmp_path / "new-cache"
    prompt = tmp_path / "a prompt file.json"
    prompt.write_text('{"prompts": [{"id": "test", "text": "text"}]}')
    argv = ["run_cache", str(tmp_path / "weights"), str(output)]
    if custom:
        argv += ["--prompts", str(prompt)]
    captured = []

    class FakeProcess:
        pid = 123
        returncode = 0

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    def start(command, **kwargs):
        captured.append(command)
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["start_new_session"] is True
        (output / "manifest.json").write_text('{"status": "complete"}')
        return FakeProcess()

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(run_cache.subprocess, "Popen", start)
    monkeypatch.setattr(run_cache.psutil, "Process", lambda pid: object())
    monkeypatch.setattr(run_cache.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 2**30))
    with pytest.raises(SystemExit) as error:
        run_cache.main()
    assert error.value.code == 0
    command, = captured
    if custom:
        assert command[-2:] == ["--prompts", str(prompt.resolve())]
    else:
        assert "--prompts" not in command
    assert json.loads((output / "watchdog.json").read_text())["status"] == "complete"


def test_missing_prompt_fails_before_output_or_subprocess(tmp_path, monkeypatch):
    output = tmp_path / "new-cache"
    monkeypatch.setattr(sys, "argv", ["run_cache", "weights", str(output), "--prompts", str(tmp_path / "missing.json")])
    monkeypatch.setattr(run_cache.subprocess, "Popen", lambda *a, **k: pytest.fail("Subprocess must not launch"))
    with pytest.raises(SystemExit) as error:
        run_cache.main()
    assert error.value.code == 2
    assert not output.exists()


def test_native_negative_is_literal_official_config_and_positive_unchanged():
    root = Path(__file__).parent
    manifest = json.loads((root / "native-prompts.json").read_text())
    source = (root / manifest["negative_prompt_source"]["local_source"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == manifest["negative_prompt_source"]["source_sha256"]
    values = [ast.literal_eval(node.value) for node in ast.parse(source).body
              if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute)
              and target.attr == "sample_neg_prompt" for target in node.targets)]
    entries = {entry["id"]: entry["text"] for entry in manifest["prompts"]}
    assert entries["native_negative"] == values[0]
    assert hashlib.sha256(entries["native_negative"].encode()).hexdigest() == "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
    original = json.loads((root / "prompts.json").read_text())
    assert entries["atrium"] == next(entry["text"] for entry in original["prompts"] if entry["id"] == "atrium")
    assert hashlib.sha256((root / "results/measured-source/run_cache.py.txt").read_bytes()).hexdigest() == "02e27782a78766b47656f0ce13bfa8ac489dd1d4d2d83bfaaccc2a56daaedca8"
