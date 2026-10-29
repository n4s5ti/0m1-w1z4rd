from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from nlcli_wizard import cli


class CommandOnlyAgent:
    def __init__(self, **kwargs):
        pass

    def translate(self, natural_language: str) -> dict:
        return {
            "command": "flow hello_world",
            "confidence": None,
            "explanation": "",
            "alternatives": [],
            "success": True,
        }


class RecordingAgent:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(self)

    def translate(self, natural_language: str) -> dict:
        self.natural_language = natural_language
        return {
            "command": "echo translated",
            "confidence": None,
            "explanation": "",
            "alternatives": [],
            "success": True,
        }


def route(name: str, model_key: str):
    return SimpleNamespace(
        requested_tool=name,
        model_key=model_key,
        executable=Path("/usr/bin/" + name),
        help_text="usage: " + name,
        allowed_prefixes=((name,),),
    )


def test_translate_displays_command_when_confidence_is_unavailable(monkeypatch):
    monkeypatch.setattr(cli, "NLCLIAgent", CommandOnlyAgent)

    result = CliRunner().invoke(
        cli.main,
        ["translate", "--cli-tool", "flow", "start", "the", "greeting", "demo"],
    )

    assert result.exit_code == 0
    assert "Command: flow hello_world" in result.output
    assert "Confidence: unavailable" in result.output
    assert "Error:" not in result.output


def test_dynamic_command_routes_arbitrary_installed_tool(monkeypatch):
    RecordingAgent.calls.clear()
    monkeypatch.setattr(cli, "NLCLIAgent", RecordingAgent)
    monkeypatch.setattr(cli, "resolve_tool_route", lambda name: route(name, "personal"))

    result = CliRunner().invoke(cli.main, ["kubectl", "show", "failing", "pods"])

    assert result.exit_code == 0
    assert RecordingAgent.calls[0].kwargs == {
        "cli_tool": "personal",
        "requested_tool": "kubectl",
        "tool_context": "usage: kubectl",
        "allowed_prefixes": (("kubectl",),),
        "model_path": None,
    }
    assert RecordingAgent.calls[0].natural_language == "show failing pods"


def test_atuin_routes_to_personal_model_but_keeps_atuin_validation(monkeypatch):
    RecordingAgent.calls.clear()
    monkeypatch.setattr(cli, "NLCLIAgent", RecordingAgent)
    monkeypatch.setattr(cli, "resolve_tool_route", lambda name: route(name, "personal"))

    result = CliRunner().invoke(cli.main, ["atuin", "search", "my", "history"])

    assert result.exit_code == 0
    assert RecordingAgent.calls[0].kwargs["cli_tool"] == "personal"
    assert RecordingAgent.calls[0].kwargs["requested_tool"] == "atuin"
    assert RecordingAgent.calls[0].kwargs["allowed_prefixes"] == (("atuin",),)


def test_docker_routes_to_dedicated_model(monkeypatch):
    RecordingAgent.calls.clear()
    monkeypatch.setattr(cli, "NLCLIAgent", RecordingAgent)
    monkeypatch.setattr(cli, "resolve_tool_route", lambda name: route(name, "docker"))

    result = CliRunner().invoke(cli.main, ["docker", "remake", "the", "dockerfile"])

    assert result.exit_code == 0
    assert RecordingAgent.calls[0].kwargs["cli_tool"] == "docker"
    assert RecordingAgent.calls[0].kwargs["requested_tool"] == "docker"
    assert RecordingAgent.calls[0].natural_language == "remake the dockerfile"


def test_unresolved_top_level_command_is_a_click_error(monkeypatch):
    from nlcli_wizard.tool_context import ToolResolutionError

    def fail(name: str):
        raise ToolResolutionError(name)

    monkeypatch.setattr(cli, "resolve_tool_route", fail)

    result = CliRunner().invoke(cli.main, ["dockre", "remake", "the", "dockerfile"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_dynamic_commands_are_not_listed_as_static_shortcuts():
    result = CliRunner().invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "translate" in result.output
    for command in ("venvy", "docker", "flow", "atuin"):
        assert command not in result.output
