import json
import os
import shlex
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

from training.build_local_curriculum import (
    build_curriculum,
    collect_help_by_tool,
    inventory_path_executables,
    main,
)


def _write_executable(path: Path, contents: str = "#!/bin/sh\nexit 0\n") -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def test_inventory_path_executables_returns_sorted_unique_executable_basenames_without_running_them(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    execution_marker = tmp_path / "executed"

    _write_executable(
        first_path / "zeta",
        "#!/bin/sh\n: > {}\n".format(shlex.quote(str(execution_marker))),
    )
    _write_executable(
        first_path / "alpha",
        "#!/bin/sh\n: > {}\n".format(shlex.quote(str(execution_marker))),
    )
    _write_executable(second_path / "alpha")
    _write_executable(first_path / "--help")
    (first_path / "ordinary-file").write_text("not executable", encoding="utf-8")
    (first_path / "directory").mkdir()

    assert inventory_path_executables(os.pathsep.join((str(first_path), str(second_path)))) == [
        "alpha",
        "zeta",
    ]
    assert not execution_marker.exists()


def test_collect_help_by_tool_runs_only_simple_resolved_tools_with_bounded_safe_subprocess_options(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("ATUIN_SESSION", "private-history-must-not-leak")
    monkeypatch.setenv("HERMES_GUIDANCE", "procedural-guidance-must-not-be-corpus")
    lookups: List[str] = []
    calls: List[Dict[str, Any]] = []

    def executable_lookup(tool: str) -> Any:
        lookups.append(tool)
        return {"git": "/isolated/bin/git", "missing": None}.get(tool)

    def command_runner(argv: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, stdout="git usage", stderr="")

    collected = collect_help_by_tool(
        ["git", "missing", "../escape", "two words", "git;rm", ""],
        executable_lookup=executable_lookup,
        command_runner=command_runner,
        timeout_seconds=0.25,
        max_bytes=1024,
    )

    assert collected == {"git": "git usage"}
    assert lookups == ["git", "missing"]
    assert len(calls) == 1
    assert calls[0]["argv"] == ["/isolated/bin/git", "--help"]
    assert calls[0]["shell"] is False
    assert calls[0]["timeout"] == 0.25
    assert calls[0]["capture_output"] is True
    assert calls[0]["text"] is True
    assert set(calls[0]["env"]).issubset({"PATH", "LANG", "LC_ALL"})
    assert "ATUIN_SESSION" not in calls[0]["env"]
    assert "HERMES_GUIDANCE" not in calls[0]["env"]


def test_collect_help_by_tool_keeps_useful_stderr_from_a_nonzero_help_exit() -> None:
    def executable_lookup(tool: str) -> Any:
        assert tool == "curl"
        return "/isolated/bin/curl"

    def command_runner(argv: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            argv,
            2,
            stdout="",
            stderr="curl: try 'curl --help' for usage",
        )

    collected = collect_help_by_tool(
        ["curl"],
        executable_lookup=executable_lookup,
        command_runner=command_runner,
        timeout_seconds=1,
        max_bytes=1024,
    )

    assert collected == {"curl": "curl: try 'curl --help' for usage"}


def test_collect_help_by_tool_truncates_captured_help_to_max_bytes() -> None:
    def executable_lookup(tool: str) -> Any:
        assert tool == "rg"
        return "/isolated/bin/rg"

    def command_runner(argv: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, 0, stdout="0123456789", stderr="")

    collected = collect_help_by_tool(
        ["rg"],
        executable_lookup=executable_lookup,
        command_runner=command_runner,
        timeout_seconds=1,
        max_bytes=4,
    )

    assert collected == {"rg": "0123"}


def test_build_curriculum_writes_a_sanitized_curriculum_and_local_basename_inventory(
    tmp_path: Path,
) -> None:
    now_ns = 1_725_000_000_000_000_000
    database_path = tmp_path / "atuin-history.db"
    output_dir = tmp_path / "curriculum"
    first_path = tmp_path / "first-bin"
    second_path = tmp_path / "second-bin"
    first_path.mkdir()
    second_path.mkdir()
    _write_executable(first_path / "zeta")
    _write_executable(first_path / "alpha")
    _write_executable(second_path / "alpha")
    _write_executable(first_path / "--help")

    with sqlite3.connect(database_path) as connection:
        connection.execute("""
            CREATE TABLE history (
                id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                duration INTEGER NOT NULL,
                exit INTEGER NOT NULL,
                command TEXT NOT NULL,
                cwd TEXT NOT NULL,
                session TEXT NOT NULL,
                hostname TEXT NOT NULL,
                deleted_at INTEGER
            )
            """)
        connection.executemany(
            """
            INSERT INTO history
                (id, timestamp, duration, exit, command, cwd, session, hostname, deleted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "private-cargo-build",
                    now_ns - 1,
                    1,
                    0,
                    "cargo build --package private-package",
                    "/private/build-cwd",
                    "session-1",
                    "localhost",
                    None,
                ),
                (
                    "private-cargo-test",
                    now_ns - 2,
                    1,
                    0,
                    "cargo test --manifest-path /private/test-manifest",
                    "/private/test-cwd",
                    "session-2",
                    "localhost",
                    None,
                ),
                (
                    "environment-command",
                    now_ns - 3,
                    1,
                    0,
                    "PRIVATE_TOKEN=secret-environment-value cargo test",
                    "/private/environment-cwd",
                    "session-3",
                    "localhost",
                    None,
                ),
                (
                    "unrequested-tool",
                    now_ns - 4,
                    1,
                    0,
                    "rg HERMES_PERSONAL_PROSE",
                    "/private/rg-cwd",
                    "session-4",
                    "localhost",
                    None,
                ),
            ],
        )

    help_calls: List[List[str]] = []
    looked_up: List[str] = []

    def executable_lookup(tool: str) -> Any:
        looked_up.append(tool)
        return {"cargo": "/isolated/bin/cargo", "git": "/isolated/bin/git"}[tool]

    def command_runner(argv: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        help_calls.append(argv)
        stdout_by_executable = {
            "/isolated/bin/cargo": "Commands:\n    build  Compile a package\n    test  Run tests\n",
            "/isolated/bin/git": "Commands:\n    status  Show repository status\n",
        }
        return subprocess.CompletedProcess(argv, 0, stdout=stdout_by_executable[argv[0]], stderr="")

    result = build_curriculum(
        db_path=database_path,
        output_dir=output_dir,
        tools=["git", "cargo"],
        path_value=os.pathsep.join((str(first_path), str(second_path))),
        now_ns=now_ns,
        window_days=30,
        minimum_count=1,
        executable_lookup=executable_lookup,
        command_runner=command_runner,
    )

    assert result == {"requested": 2, "collected": 2, "emitted": 8}
    assert looked_up == ["git", "cargo"]
    assert help_calls == [
        ["/isolated/bin/git", "--help"],
        ["/isolated/bin/cargo", "--help"],
    ]
    assert (output_dir / "train.jsonl").is_file()
    assert (output_dir / "test.jsonl").is_file()
    assert (output_dir / "manifest.json").is_file()
    train_rows = [
        json.loads(line)
        for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    test_rows = [
        json.loads(line)
        for line in (output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows = train_rows + test_rows
    expected_description_by_output = {
        "COMMAND: cargo build\n": "Compile a package",
        "COMMAND: cargo test\n": "Run tests",
    }
    train_outputs = {row["output"] for row in train_rows}
    test_outputs = {row["output"] for row in test_rows}
    assert train_outputs.isdisjoint(test_outputs)
    assert train_outputs | test_outputs == set(expected_description_by_output)
    assert Counter(row["output"] for row in rows) == {
        output: 4 for output in expected_description_by_output
    }
    for row in rows:
        target_command = row["output"].removeprefix("COMMAND: ").strip()
        assert target_command not in row["instruction"]
    assert json.loads((output_dir / "installed_cli_inventory.json").read_text()) == {
        "count": 2,
        "executables": ["alpha", "zeta"],
    }
    assert json.loads((output_dir / "coverage_gaps.json").read_text()) == {
        "covered_count": 1,
        "covered_tools": ["cargo"],
        "installed_count": 2,
        "pending_count": 2,
        "pending_tools": ["alpha", "zeta"],
    }

    emitted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            output_dir / "train.jsonl",
            output_dir / "test.jsonl",
            output_dir / "manifest.json",
            output_dir / "installed_cli_inventory.json",
            output_dir / "coverage_gaps.json",
        )
    )
    for private_value in (
        "cargo build --package private-package",
        "cargo test --manifest-path /private/test-manifest",
        "PRIVATE_TOKEN=secret-environment-value cargo test",
        "rg HERMES_PERSONAL_PROSE",
        "private-package",
        "/private/build-cwd",
        "/private/test-cwd",
        "/private/environment-cwd",
        "/private/rg-cwd",
        "PRIVATE_TOKEN",
        "secret-environment-value",
        "HERMES_PERSONAL_PROSE",
    ):
        assert private_value not in emitted


def test_main_requires_at_least_one_explicit_tool(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "--db",
                str(tmp_path / "atuin-history.db"),
                "--output",
                str(tmp_path / "curriculum"),
            ]
        )

    assert raised.value.code == 2


def test_main_forwards_repeatable_tool_and_curriculum_options(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    captured: Dict[str, Any] = {}

    def fake_build_curriculum(**kwargs: Any) -> Dict[str, int]:
        captured.update(kwargs)
        return {"requested": 2, "collected": 2, "emitted": 2}

    monkeypatch.setattr(
        "training.build_local_curriculum.build_curriculum",
        fake_build_curriculum,
    )

    main(
        [
            "--db",
            str(tmp_path / "atuin-history.db"),
            "--output",
            str(tmp_path / "curriculum"),
            "--window-days",
            "14",
            "--minimum-count",
            "3",
            "--tool",
            "cargo",
            "--tool",
            "git",
        ]
    )

    assert captured["db_path"] == tmp_path / "atuin-history.db"
    assert captured["output_dir"] == tmp_path / "curriculum"
    assert captured["tools"] == ["cargo", "git"]
    assert captured["window_days"] == 14
    assert captured["minimum_count"] == 3
