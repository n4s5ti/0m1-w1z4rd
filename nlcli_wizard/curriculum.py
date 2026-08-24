"""Build a privacy-preserving local CLI curriculum from Atuin history."""

import hashlib
import json
import re
import shlex
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set, Tuple, Union

_SIMPLE_EXECUTABLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_ENVIRONMENT_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
_ASSIGNMENT_IN_TEXT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*\S+")
_HELP_COLUMNS = re.compile(r"\s{2,}")
_NON_COMMAND_HELP_SECTIONS = frozenset(
    {
        "additional help topics",
        "authentication",
        "command chaining",
        "configuration",
        "environment",
        "examples",
        "install",
        "options",
        "snapshot options",
    }
)
_ABSOLUTE_PATH = re.compile(r"(?:~[/\\]|(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9_.-]+/)+[^\s\"']*)")
_SECRET = re.compile(
    r"(?:\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\b\s*[:=]"
    r"|\b(?:secret|token|key|password)[_-][A-Za-z0-9_-]{4,}"
    r"|\b(?:ghp|github_pat|sk|AKIA)[_-][A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)


PathLike = Union[str, Path]


def _help_descriptions(help_text: str, tool: str = "") -> Dict[str, str]:
    """Return entries only from command-bearing help sections."""
    descriptions: Dict[str, str] = {}
    command_section = False
    for line in help_text.splitlines():
        stripped = line.strip()
        if line == stripped and stripped.endswith(":"):
            section = stripped[:-1].strip().lower()
            command_section = section not in _NON_COMMAND_HELP_SECTIONS
            continue
        if not command_section:
            continue

        columns = _HELP_COLUMNS.split(stripped, maxsplit=1)
        if len(columns) != 2:
            continue
        usage, description = columns
        command = usage.split(maxsplit=1)[0].rstrip(",")
        if (
            _SIMPLE_EXECUTABLE.fullmatch(command) is not None
            and command[:1].islower()
            and command != tool
        ):
            descriptions[command] = description.strip()
    return descriptions


def _unsafe_text(value: str) -> bool:
    return bool(
        _ABSOLUTE_PATH.search(value) or _SECRET.search(value) or _ASSIGNMENT_IN_TEXT.search(value)
    )


def _jsonl(rows: Iterable[Dict[str, str]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=False) + "\n"
        for row in rows
    )


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _instruction_variants(tool: str, description: str) -> Tuple[str, ...]:
    lowered = description[:1].lower() + description[1:]
    prefix = "Translate to personal command: "
    variants = [
        f"{prefix}{description}",
        f"{prefix}{tool}: {description}",
        f"{prefix}Use {tool} to {lowered}",
        f"{prefix}Please {lowered} using {tool}",
    ]
    without_parenthetical = " ".join(re.sub(r"\s*\([^()]*\)", "", description).split())
    if without_parenthetical != description:
        variants.append(f"{prefix}{without_parenthetical}")
    naturalized = re.sub(
        r"\byour history\b",
        "my shell history",
        without_parenthetical,
        flags=re.IGNORECASE,
    )
    naturalized = re.sub(
        r"\bin an interactive UI\b",
        "interactively",
        naturalized,
        flags=re.IGNORECASE,
    )
    if naturalized not in {description, without_parenthetical}:
        variants.append(f"{prefix}{naturalized}")
    return tuple(variants)


def _read_history(db_path: PathLike) -> Iterable[Tuple[int, str, str, str, object]]:
    database_uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        cursor = connection.execute("""
            SELECT timestamp, command, cwd, session, deleted_at
            FROM history
            """)
        yield from cursor


def build_local_curriculum(
    db_path: PathLike,
    help_by_tool: Mapping[str, str],
    approved_tools: Set[str],
    output_dir: PathLike,
    now_ns: int,
    window_days: int,
    minimum_count: int,
) -> None:
    """Write sanitized, aggregate-derived train/test JSONL curriculum files.

    Raw Atuin rows are consumed only while aggregating command skeleton counts and
    never appear in the resulting JSONL or manifest.
    """
    if window_days < 0 or minimum_count < 1:
        raise ValueError("window_days and minimum_count must be positive")

    cutoff_ns = now_ns - window_days * 24 * 60 * 60 * 1_000_000_000
    descriptions = {
        tool: _help_descriptions(help_text, tool=tool)
        for tool, help_text in help_by_tool.items()
        if _SIMPLE_EXECUTABLE.fullmatch(tool) is not None
    }
    exclusions = Counter()
    skeleton_counts = Counter()
    skeleton_sessions: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    tool_usage_counts = Counter()
    tool_usage_sessions: Dict[str, Set[str]] = defaultdict(set)

    for timestamp, command, cwd, session, deleted_at in _read_history(db_path):
        if deleted_at is not None:
            exclusions["deleted"] += 1
            continue
        if timestamp < cutoff_ns or timestamp > now_ns:
            exclusions["out_of_window"] += 1
            continue
        if not cwd or cwd == "unknown":
            exclusions["invalid_cwd"] += 1
            continue
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            exclusions["unparseable_command"] += 1
            continue
        if not tokens:
            exclusions["unsupported_command_shape"] += 1
            continue
        if _ENVIRONMENT_ASSIGNMENT.fullmatch(tokens[0]) is not None:
            exclusions["environment_assignment"] += 1
            continue

        if any(
            token in {";", "|", "||", "&", "&&", "<", ">", ">>", "<<", "`"}
            or any(marker in token for marker in (";", "|", "&", "<", ">", "$(", "`"))
            for token in tokens[1:]
        ):
            exclusions["unsupported_command_shape"] += 1
            continue
        executable = tokens[0]
        if (
            _SIMPLE_EXECUTABLE.fullmatch(executable) is None
            or executable not in approved_tools
            or executable not in descriptions
        ):
            exclusions["unapproved_tool"] += 1
            continue
        tool_usage_counts[executable] += 1
        tool_usage_sessions[executable].add(session)

        subcommand = next((token for token in tokens[1:] if not token.startswith("-")), None)
        if subcommand is None or subcommand not in descriptions[executable]:
            exclusions["unsupported_command_shape"] += 1
            continue
        if _SIMPLE_EXECUTABLE.fullmatch(subcommand) is None:
            exclusions["unsupported_command_shape"] += 1
            continue

        description = descriptions[executable][subcommand]
        if _unsafe_text(description):
            exclusions["unsafe_help_description"] += 1
            continue

        skeleton = (executable, subcommand)
        skeleton_counts[skeleton] += 1
        skeleton_sessions[skeleton].add(session)

    rows = []
    aggregate_skeletons = {}
    tool_counts = Counter()
    category_counts = Counter()
    for tool in sorted(tool_usage_counts):
        usage_count = tool_usage_counts[tool]
        if usage_count < minimum_count:
            exclusions["below_minimum_count"] += usage_count
            continue

        for subcommand, description in sorted(descriptions[tool].items()):
            if _SIMPLE_EXECUTABLE.fullmatch(subcommand) is None:
                continue
            if _unsafe_text(description):
                exclusions["unsafe_help_description"] += 1
                continue

            skeleton = (tool, subcommand)
            output = "COMMAND: {} {}\n".format(tool, subcommand)
            variant_rows = [
                {
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "category": tool,
                }
                for instruction in _instruction_variants(tool, description)
            ]
            rows.append(("{} {}".format(tool, subcommand), variant_rows))
            aggregate_skeletons["{} {}".format(tool, subcommand)] = {
                "count": skeleton_counts[skeleton],
                "sessions": len(skeleton_sessions[skeleton]),
            }

        tool_counts[tool] = usage_count
        category_counts[tool] = usage_count

    train_rows = []
    test_rows = []
    for skeleton, variant_rows in rows:
        target = (
            test_rows if hashlib.sha256(skeleton.encode("utf-8")).digest()[0] < 51 else train_rows
        )
        target.extend(variant_rows)

    train_content = _jsonl(train_rows)
    test_content = _jsonl(test_rows)
    manifest = {
        "raw_history_persisted": False,
        "exclusions": dict(sorted(exclusions.items())),
        "aggregates": {
            "skeletons": aggregate_skeletons,
            "tools": dict(sorted(tool_counts.items())),
            "tool_sessions": {tool: len(tool_usage_sessions[tool]) for tool in sorted(tool_counts)},
            "categories": dict(sorted(category_counts.items())),
        },
        "content_digests": {
            "train.jsonl": _content_digest(train_content),
            "test.jsonl": _content_digest(test_content),
        },
    }
    manifest_content = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"

    emitted_payload = train_content + test_content + manifest_content
    if _unsafe_text(emitted_payload):
        raise ValueError("unsafe curriculum payload")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "train.jsonl").write_text(train_content, encoding="utf-8")
    (destination / "test.jsonl").write_text(test_content, encoding="utf-8")
    (destination / "manifest.json").write_text(manifest_content, encoding="utf-8")
