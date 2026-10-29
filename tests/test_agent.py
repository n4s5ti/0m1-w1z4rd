from nlcli_wizard.agent import NLCLIAgent


class CommandOnlyModel:
    def __init__(self, command: str = "flow hello_world"):
        self.command = command

    def generate_command(self, natural_language: str) -> str:
        return f"COMMAND: {self.command}"


def test_command_only_model_output_is_successful():
    agent = NLCLIAgent("flow")
    agent.model_manager = CommandOnlyModel()

    result = agent.translate("start the greeting demo")

    assert result == {
        "command": "flow hello_world",
        "argv": ["flow", "hello_world"],
        "confidence": None,
        "explanation": "",
        "alternatives": [],
        "success": True,
    }


def test_validation_requires_an_exact_allowed_prefix():
    agent = NLCLIAgent(
        "docker",
        requested_tool="docker",
        allowed_prefixes=(("docker",),),
    )

    assert agent._validate_command("docker ps")
    assert not agent._validate_command("dockerx ps")


def test_personal_model_cannot_validate_another_executable():
    agent = NLCLIAgent(
        "personal",
        requested_tool="atuin",
        allowed_prefixes=(("atuin",),),
    )

    assert agent._validate_command("atuin search shell")
    assert not agent._validate_command("docker ps")


def test_validation_rejects_shell_operators_and_malformed_argv():
    agent = NLCLIAgent(
        "docker",
        requested_tool="docker",
        allowed_prefixes=(("docker",),),
    )

    assert not agent._validate_command("docker ps && rm -rf /")
    assert not agent._validate_command("docker run > output.txt")
    assert not agent._validate_command("docker 'unterminated")


def test_translate_normalizes_command_from_validated_argv():
    agent = NLCLIAgent("flow")
    agent.model_manager = CommandOnlyModel('flow hello_world --name "Ada Lovelace"')

    result = agent.translate("start the greeting demo")

    assert result["argv"] == ["flow", "hello_world", "--name", "Ada Lovelace"]
    assert result["command"] == "flow hello_world --name 'Ada Lovelace'"


def test_translate_does_not_return_invalid_command_or_argv():
    agent = NLCLIAgent(
        "docker",
        requested_tool="docker",
        allowed_prefixes=(("docker",),),
    )
    agent.model_manager = CommandOnlyModel("docker ps && rm -rf /")

    result = agent.translate("list containers")

    assert result["command"] == ""
    assert result["argv"] == []
    assert result["success"] is False
