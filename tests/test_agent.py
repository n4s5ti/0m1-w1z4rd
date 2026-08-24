from nlcli_wizard.agent import NLCLIAgent


class CommandOnlyModel:
    def generate_command(self, natural_language: str) -> str:
        return "COMMAND: flow hello_world"


def test_command_only_model_output_is_successful():
    agent = NLCLIAgent("flow")
    agent.model_manager = CommandOnlyModel()

    result = agent.translate("start the greeting demo")

    assert result == {
        "command": "flow hello_world",
        "confidence": None,
        "explanation": "",
        "alternatives": [],
        "success": True,
    }
