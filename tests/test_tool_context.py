import os
import subprocess
from types import SimpleNamespace

import pytest

from nlcli_wizard import tool_context


@pytest.fixture(autouse=True)
def clear_help_cache():
    tool_context._HELP_CACHE.clear()


def _fake_completed_process(output=""):
    return SimpleNamespace(stdout=output)


@pytest.mark.parametrize(
    ("name", "expected_model"),
    [
        ("docker", "docker"),
        ("venvy", "venvy"),
        ("flow", "flow"),
        ("atuin", "personal"),
    ],
)
def test_known_tools_use_their_declared_model_route(monkeypatch, tmp_path, name, expected_model):
    executable = tmp_path / name
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(tool_context.shutil, "which", lambda requested: str(executable))
    monkeypatch.setattr(
        tool_context.subprocess,
        "run",
        lambda *args, **kwargs: _fake_completed_process("safe reference"),
    )

    route = tool_context.resolve_tool_route(name)

    assert route.requested_tool == name
    assert route.model_key == expected_model
    assert route.executable == executable
    assert route.help_text == "safe reference"
    assert route.allowed_prefixes == ((name,),)


def test_arbitrary_installed_executable_uses_personal_model(monkeypatch, tmp_path):
    executable = tmp_path / "rg"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(tool_context.shutil, "which", lambda requested: str(executable))
    monkeypatch.setattr(
        tool_context.subprocess,
        "run",
        lambda *args, **kwargs: _fake_completed_process("reference"),
    )

    route = tool_context.resolve_tool_route("rg")

    assert route.model_key == "personal"
    assert route.allowed_prefixes == (("rg",),)


@pytest.mark.parametrize(
    "name", ["", "../docker", "docker/run", "docker tool", "docker;id", "-docker"]
)
def test_invalid_tool_tokens_are_rejected(name):
    with pytest.raises(tool_context.ToolResolutionError):
        tool_context.resolve_tool_route(name)


def test_unresolved_tool_is_rejected(monkeypatch):
    monkeypatch.setattr(tool_context.shutil, "which", lambda requested: None)

    with pytest.raises(tool_context.ToolResolutionError, match="not found"):
        tool_context.resolve_tool_route("dockre")


def test_help_collection_invokes_only_resolved_executable_with_help(monkeypatch, tmp_path):
    executable = tmp_path / "docker"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    calls = []
    monkeypatch.setattr(tool_context.shutil, "which", lambda requested: str(executable))

    def record_run(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_completed_process("reference")

    monkeypatch.setattr(tool_context.subprocess, "run", record_run)

    tool_context.resolve_tool_route("docker")

    assert calls == [
        (
            ([str(executable), "--help"],),
            {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "errors": "replace",
                "timeout": tool_context.HELP_TIMEOUT_SECONDS,
                "check": False,
            },
        )
    ]


@pytest.mark.parametrize(
    "failure", [subprocess.TimeoutExpired(["docker", "--help"], 1), OSError("denied")]
)
def test_help_timeout_or_failure_returns_empty_inert_text(monkeypatch, tmp_path, failure):
    executable = tmp_path / "docker"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(tool_context.shutil, "which", lambda requested: str(executable))

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(tool_context.subprocess, "run", fail)

    route = tool_context.resolve_tool_route("docker")

    assert route.help_text == ""


def test_help_text_is_strictly_bounded_to_maximum_characters(monkeypatch, tmp_path):
    executable = tmp_path / "docker"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(tool_context.shutil, "which", lambda requested: str(executable))
    monkeypatch.setattr(
        tool_context.subprocess,
        "run",
        lambda *args, **kwargs: _fake_completed_process("x" * (tool_context.MAX_HELP_CHARS + 1)),
    )

    route = tool_context.resolve_tool_route("docker")

    assert route.help_text == "x" * tool_context.MAX_HELP_CHARS
    assert len(route.help_text) == tool_context.MAX_HELP_CHARS


def test_executable_replacement_invalidates_cached_help(monkeypatch, tmp_path):
    executable = tmp_path / "docker"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setattr(tool_context.shutil, "which", lambda requested: str(executable))
    calls = []

    def record_run(*args, **kwargs):
        calls.append(None)
        return _fake_completed_process("first" if len(calls) == 1 else "second")

    monkeypatch.setattr(tool_context.subprocess, "run", record_run)

    assert tool_context.resolve_tool_route("docker").help_text == "first"
    executable.unlink()
    executable.write_text("#!/bin/sh\n# replaced\n")
    executable.chmod(0o755)
    os.utime(executable, None)

    assert tool_context.resolve_tool_route("docker").help_text == "second"
    assert len(calls) == 2
