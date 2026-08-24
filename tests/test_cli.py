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


def test_docker_shortcut_routes_joined_instruction_to_docker(monkeypatch):
    RecordingAgent.calls.clear()
    monkeypatch.setattr(cli, "NLCLIAgent", RecordingAgent)

    result = CliRunner().invoke(cli.main, ["docker", "remake", "the", "dockerfile"])

    assert result.exit_code == 0
    assert RecordingAgent.calls[0].kwargs["cli_tool"] == "docker"
    assert RecordingAgent.calls[0].natural_language == "remake the dockerfile"


def test_atuin_shortcut_routes_unchanged_instruction_to_personal(monkeypatch):
    RecordingAgent.calls.clear()
    monkeypatch.setattr(cli, "NLCLIAgent", RecordingAgent)

    result = CliRunner().invoke(cli.main, ["atuin", "search", "my", "history"])

    assert result.exit_code == 0
    assert RecordingAgent.calls[0].kwargs["cli_tool"] == "personal"
    assert RecordingAgent.calls[0].natural_language == "search my history"


def test_shortcuts_are_visible_in_help():
    result = CliRunner().invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    for command in ("venvy", "docker", "flow", "atuin"):
        assert command in result.output


def test_unknown_top_level_command_is_rejected():
    result = CliRunner().invoke(cli.main, ["dockre", "remake", "the", "dockerfile"])

    assert result.exit_code != 0
    assert "No such command" in result.output
