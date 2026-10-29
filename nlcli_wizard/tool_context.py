"""Resolve a PATH executable into bounded, untrusted model reference context."""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

MAX_HELP_CHARS = 1600
HELP_TIMEOUT_SECONDS = 1.0

_MODEL_KEYS = {
    "venvy": "venvy",
    "docker": "docker",
    "flow": "flow",
    "atuin": "personal",
}
_SAFE_COMMAND_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_HELP_CACHE: Dict[Tuple[str, int, int, int, int], str] = {}


class ToolResolutionError(ValueError):
    """Raised when a requested tool cannot be safely resolved from PATH."""


@dataclass(frozen=True)
class ToolRoute:
    """The model route and bounded reference data for one executable."""

    requested_tool: str
    model_key: str
    executable: Path
    help_text: str
    allowed_prefixes: Tuple[Tuple[str, ...], ...]


def _cache_key(executable: Path) -> Tuple[str, int, int, int, int]:
    stat_result = executable.stat()
    return (
        str(executable),
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mtime_ns,
        stat_result.st_size,
    )


def _collect_help(executable: Path) -> str:
    """Return bounded help text; command failures intentionally yield no context."""

    cache_key = _cache_key(executable)
    cached_help = _HELP_CACHE.get(cache_key)
    if cached_help is not None:
        return cached_help

    try:
        result = subprocess.run(
            [str(executable), "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=HELP_TIMEOUT_SECONDS,
            check=False,
        )
        help_text = (result.stdout or "")[:MAX_HELP_CHARS]
    except (OSError, subprocess.TimeoutExpired):
        help_text = ""

    _HELP_CACHE[cache_key] = help_text
    return help_text


def resolve_tool_route(name: str) -> ToolRoute:
    """Resolve one safe PATH command token without executing user-generated commands."""

    if not isinstance(name, str) or not _SAFE_COMMAND_NAME.fullmatch(name):
        raise ToolResolutionError("tool name must be a single safe command token")

    resolved_path = shutil.which(name)
    if resolved_path is None:
        raise ToolResolutionError("tool {!r} was not found on PATH".format(name))

    executable = Path(resolved_path).resolve()
    try:
        help_text = _collect_help(executable)
    except OSError as error:
        raise ToolResolutionError("tool {!r} could not be inspected".format(name)) from error

    return ToolRoute(
        requested_tool=name,
        model_key=_MODEL_KEYS.get(name, "personal"),
        executable=executable,
        help_text=help_text,
        allowed_prefixes=((name,),),
    )
