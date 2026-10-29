from nlcli_wizard.agent import NLCLIAgent
from nlcli_wizard.model import ModelManager


def _agent_for_validation(model_key, requested_tool=None):
    agent = object.__new__(NLCLIAgent)
    agent.cli_tool = model_key
    agent.requested_tool = requested_tool or model_key
    agent.allowed_prefixes = ((agent.requested_tool,),)
    return agent


def test_personal_registry_entry_is_local_only_with_personal_model_filename():
    assert ModelManager.MODEL_REGISTRY["personal"] == {
        "filename": "wiz4rd-personal.Q4_K_M.gguf",
        "repo": None,
    }


def test_personal_prompt_contains_the_exact_training_user_turn():
    manager = object.__new__(ModelManager)
    manager.cli_tool = "personal"
    manager.requested_tool = "personal"
    manager.tool_context = ""

    prompt = manager._build_prompt("compile the package")

    assert (
        "<start_of_turn>user\n" "Translate to personal command: compile the package<end_of_turn>\n"
    ) in prompt


def test_personal_model_accepts_the_requested_executable_prefix():
    agent = _agent_for_validation("personal", requested_tool="cargo")

    assert agent._validate_command("cargo build") is True


def test_personal_model_rejects_another_executable_prefix():
    agent = _agent_for_validation("personal", requested_tool="cargo")

    assert agent._validate_command("git status") is False


def test_personal_validation_rejects_shell_injection():
    agent = _agent_for_validation("personal", requested_tool="cargo")

    assert agent._validate_command("cargo build; echo compromised") is False


def test_tool_specific_validation_accepts_its_named_prefix():
    agent = _agent_for_validation("docker")

    assert agent._validate_command("docker ps") is True


def test_tool_specific_validation_rejects_another_executable_prefix():
    agent = _agent_for_validation("docker")

    assert agent._validate_command("cargo build") is False
