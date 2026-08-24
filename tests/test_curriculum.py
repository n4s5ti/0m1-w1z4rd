import json
import sqlite3
from collections import Counter

from nlcli_wizard.curriculum import (
    _help_descriptions,
    _instruction_variants,
    build_local_curriculum,
)


def test_build_local_curriculum_emits_only_sanitized_approved_help_validated_rows(tmp_path):
    now_ns = 1_725_000_000_000_000_000
    database_path = tmp_path / "history.db"
    output_dir = tmp_path / "local-curriculum"

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
                    "cargo-test-private-argument",
                    now_ns - 1,
                    1,
                    0,
                    "cargo test --package private-name",
                    "/work/private-cwd",
                    "session-1",
                    "localhost",
                    None,
                ),
                (
                    "cargo-build-private-path",
                    now_ns - 2,
                    1,
                    0,
                    "cargo build /home/alice/secret",
                    "/work/another-private-cwd",
                    "session-1",
                    "localhost",
                    None,
                ),
                (
                    "cargo-test-environment-token",
                    now_ns - 3,
                    1,
                    0,
                    "PRIVATE_TOKEN=top-secret-token cargo test",
                    "/work/token-cwd",
                    "session-1",
                    "localhost",
                    None,
                ),
                (
                    "unapproved-git-command",
                    now_ns - 4,
                    1,
                    0,
                    "git status",
                    "/work/rejected-tool-cwd",
                    "session-1",
                    "localhost",
                    None,
                ),
                (
                    "cargo-test-unknown-cwd",
                    now_ns - 5,
                    1,
                    0,
                    "cargo test --doc",
                    "unknown",
                    "session-1",
                    "localhost",
                    None,
                ),
            ],
        )

    build_local_curriculum(
        db_path=database_path,
        help_by_tool={
            "cargo": """Usage: cargo <COMMAND>\n\nCommands:\n    build    Compile the current package\n    clean    Remove generated artifacts\n    test     Execute unit and integration tests\n"""
        },
        approved_tools={"cargo"},
        output_dir=output_dir,
        now_ns=now_ns,
        window_days=90,
        minimum_count=1,
    )

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    manifest_path = output_dir / "manifest.json"
    assert train_path.is_file()
    assert test_path.is_file()
    assert manifest_path.is_file()

    train_rows = [json.loads(line) for line in train_path.read_text().splitlines() if line]
    test_rows = [json.loads(line) for line in test_path.read_text().splitlines() if line]
    rows = train_rows + test_rows
    expected_description_by_output = {
        "COMMAND: cargo build\n": "Compile the current package",
        "COMMAND: cargo clean\n": "Remove generated artifacts",
        "COMMAND: cargo test\n": "Execute unit and integration tests",
    }
    train_outputs = {row["output"] for row in train_rows}
    test_outputs = {row["output"] for row in test_rows}
    assert train_outputs.isdisjoint(test_outputs)
    assert train_outputs | test_outputs == set(expected_description_by_output)
    assert Counter(row["output"] for row in rows) == {
        output: 4 for output in expected_description_by_output
    }
    for output, description in expected_description_by_output.items():
        assert "Translate to personal command: {}".format(description) in {
            row["instruction"] for row in rows if row["output"] == output
        }
    for row in rows:
        assert set(row) == {"instruction", "input", "output", "category"}
        target_command = row["output"].removeprefix("COMMAND: ").strip()
        assert target_command not in row["instruction"]
        assert row["input"] == ""
        assert row["output"].startswith("COMMAND: cargo ")
        assert row["output"].endswith("\n")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["raw_history_persisted"] is False
    assert manifest["exclusions"]["environment_assignment"] == 1
    assert manifest["exclusions"]["unapproved_tool"] == 1

    emitted = "\n".join(path.read_text() for path in (train_path, test_path, manifest_path))
    for sensitive_or_rejected_value in (
        "--package",
        "--doc",
        "private-name",
        "/home/alice/secret",
        "PRIVATE_TOKEN",
        "top-secret-token",
        "/work/private-cwd",
        "/work/another-private-cwd",
        "/work/token-cwd",
        "/work/rejected-tool-cwd",
        "unknown",
        "git status",
    ):
        assert sensitive_or_rejected_value not in emitted


def test_help_parser_ignores_environment_and_option_tables():
    help_text = """Core Commands:
  open <url>                 Navigate to URL
  agent-browser skills get    Load a skill
  build, b                   Compile the current package
  snapshot                   Accessibility tree with refs
Additional help topics:
  buildjson                  Build event JSON schema

Options:
  --json                     JSON output

Environment:
  AGENT_BROWSER_ALLOWED_DOMAINS  Restrict network domains
  AI_GATEWAY_API_KEY             API key for the AI gateway
  NO_PROXY                       Bypass proxy for hosts
"""

    assert _help_descriptions(help_text, tool="agent-browser") == {
        "open": "Navigate to URL",
        "build": "Compile the current package",
        "snapshot": "Accessibility tree with refs",
    }


def test_instruction_variants_include_help_description_without_parenthetical_detail():
    variants = _instruction_variants(
        "atuin",
        "Print the default atuin configuration (config.toml)",
    )

    assert "Translate to personal command: Print the default atuin configuration" in variants


def test_instruction_variants_naturalize_shell_history_phrasing():
    variants = _instruction_variants(
        "atuin",
        "Search your history in an interactive UI",
    )

    assert "Translate to personal command: Search my shell history interactively" in variants
