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
