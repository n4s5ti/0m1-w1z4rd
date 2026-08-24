from types import SimpleNamespace

import nlcli_wizard.agent as agent_module
from nlcli_wizard.agent import NLCLIAgent
from nlcli_wizard.model import ModelManager


def _agent_for_validation(cli_tool):
    agent = object.__new__(NLCLIAgent)
    agent.cli_tool = cli_tool
    return agent


def test_personal_registry_entry_is_local_only_with_personal_model_filename():
    assert ModelManager.MODEL_REGISTRY["personal"] == {
        "filename": "wiz4rd-personal.Q4_K_M.gguf",
        "repo": None,
    }


def test_personal_prompt_contains_the_exact_training_user_turn():
    manager = object.__new__(ModelManager)
    manager.cli_tool = "personal"

    prompt = manager._build_prompt("compile the package")

    assert (
        "<start_of_turn>user\n" "Translate to personal command: compile the package<end_of_turn>\n"
    ) in prompt


def test_personal_validation_accepts_an_installed_command_prefix(monkeypatch):
    lookups = []

    def resolve_executable(executable):
        lookups.append(executable)
        return "/mock/bin/cargo" if executable == "cargo" else None

    monkeypatch.setattr(
        agent_module,
        "shutil",
        SimpleNamespace(which=resolve_executable),
        raising=False,
    )
    agent = _agent_for_validation("personal")

    assert agent._validate_command("cargo build") is True
    assert lookups == ["cargo"]


def test_personal_validation_rejects_a_command_with_no_installed_executable(monkeypatch):
    lookups = []

    def resolve_executable(executable):
        lookups.append(executable)
        return None

    monkeypatch.setattr(
        agent_module,
        "shutil",
        SimpleNamespace(which=resolve_executable),
        raising=False,
    )
    agent = _agent_for_validation("personal")

    assert agent._validate_command("missing-command run") is False
    assert lookups == ["missing-command"]


def test_personal_validation_rejects_shell_injection_before_executable_lookup(monkeypatch):
    def unexpected_lookup(executable):
        raise AssertionError(f"shell injection reached executable lookup: {executable}")

    monkeypatch.setattr(
        agent_module,
        "shutil",
        SimpleNamespace(which=unexpected_lookup),
        raising=False,
    )
    agent = _agent_for_validation("personal")

    assert agent._validate_command("cargo build; echo compromised") is False


def test_tool_specific_validation_accepts_its_named_prefix():
    agent = _agent_for_validation("docker")

    assert agent._validate_command("docker ps") is True


def test_tool_specific_validation_rejects_another_executable_prefix():
    agent = _agent_for_validation("docker")

    assert agent._validate_command("cargo build") is False
