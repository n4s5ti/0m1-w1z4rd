"""Collect local command help for personal curriculum construction."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

_SIMPLE_TOOL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")


def inventory_path_executables(path_value: str) -> list:
    """Return sorted, unique executable names present in PATH directories."""
    executables = set()
    for directory in path_value.split(os.pathsep):
        directory = directory or os.curdir
        try:
            entries = os.listdir(directory)
        except OSError:
            continue

        for name in entries:
            candidate = os.path.join(directory, name)
            if (
                _SIMPLE_TOOL_NAME.fullmatch(name)
                and os.path.isfile(candidate)
                and os.access(candidate, os.X_OK)
            ):
                executables.add(name)
    return sorted(executables)


def collect_help_by_tool(
    tools: Iterable[str],
    executable_lookup: Callable[[str], Optional[str]] = shutil.which,
    command_runner: Callable = subprocess.run,
    timeout_seconds: float = 5.0,
    max_bytes: int = 262144,
) -> Dict[str, str]:
    """Collect bounded help text for explicitly named, resolvable tools."""
    collected = {}
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
    }

    for tool in tools:
        if not isinstance(tool, str) or not _SIMPLE_TOOL_NAME.fullmatch(tool):
            continue
        if tool in collected:
            continue

        try:
            resolved = executable_lookup(tool)
        except OSError:
            continue
        if not resolved:
            continue

        try:
            completed = command_runner(
                [resolved, "--help"],
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue

        output = completed.stdout or completed.stderr
        if not output:
            continue
        if isinstance(output, bytes):
            output_bytes = output
        else:
            output_bytes = str(output).encode("utf-8", errors="replace")
        help_text = output_bytes[: max(0, max_bytes)].decode("utf-8", errors="replace")
        if help_text:
            collected[tool] = help_text

    return collected


def build_curriculum(
    db_path: str | Path | None,
    output_dir: str | Path,
    tools: Iterable[str],
    path_value: Optional[str] = None,
    now_ns: Optional[int] = None,
    window_days: int = 30,
    minimum_count: int = 1,
    executable_lookup: Callable[[str], Optional[str]] = shutil.which,
    command_runner: Callable = subprocess.run,
) -> Dict[str, int]:
    """Build a sanitized curriculum from explicit tools and local history."""
    from nlcli_wizard import curriculum

    requested_tools = list(tools)
    help_by_tool = collect_help_by_tool(
        requested_tools,
        executable_lookup=executable_lookup,
        command_runner=command_runner,
    )
    destination = Path(output_dir)
    curriculum.build_local_curriculum(
        db_path=(
            db_path
            if db_path is not None
            else Path.home() / ".local" / "share" / "atuin" / "history.db"
        ),
        help_by_tool=help_by_tool,
        approved_tools=set(requested_tools),
        output_dir=destination,
        now_ns=time.time_ns() if now_ns is None else now_ns,
        window_days=window_days,
        minimum_count=minimum_count,
    )

    executables = inventory_path_executables(
        os.environ.get("PATH", "") if path_value is None else path_value
    )
    inventory = {"count": len(executables), "executables": executables}
    (destination / "installed_cli_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    covered_tools = sorted(manifest["aggregates"]["tools"])
    covered_set = set(covered_tools)
    pending_tools = [name for name in executables if name not in covered_set]
    coverage = {
        "covered_count": len(covered_tools),
        "covered_tools": covered_tools,
        "installed_count": len(executables),
        "pending_count": len(pending_tools),
        "pending_tools": pending_tools,
    }
    (destination / "coverage_gaps.json").write_text(
        json.dumps(coverage, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    emitted = 0
    for filename in ("train.jsonl", "test.jsonl"):
        with (destination / filename).open(encoding="utf-8") as rows:
            emitted += sum(1 for line in rows if line.strip() and json.loads(line))

    return {
        "requested": len(requested_tools),
        "collected": len(help_by_tool),
        "emitted": emitted,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Build a local curriculum using only explicitly requested tools."""
    parser = argparse.ArgumentParser(description="Build a local CLI curriculum")
    parser.add_argument("--tool", action="append", required=True)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path.home() / ".local" / "share" / "atuin" / "history.db",
    )
    parser.add_argument("--output", type=Path, default=Path("data/personal-curriculum"))
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--minimum-count", type=int, default=1)
    args = parser.parse_args(argv)
    build_curriculum(
        db_path=args.db,
        output_dir=args.output,
        tools=args.tool,
        window_days=args.window_days,
        minimum_count=args.minimum_count,
    )


if __name__ == "__main__":
    main()
