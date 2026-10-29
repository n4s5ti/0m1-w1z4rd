from nlcli_wizard.model import ModelManager


def test_prompt_names_requested_executable_separately_from_model_key():
    manager = ModelManager(
        "personal",
        requested_tool="atuin",
        tool_context="atuin search shell\n--limit NUMBER",
    )

    prompt = manager._build_prompt("show recent shell commands")

    assert "Translate to personal command for requested executable atuin:" in prompt
    assert "BEGIN UNTRUSTED CLI REFERENCE DATA" in prompt
    assert "atuin search shell" in prompt
    assert "END UNTRUSTED CLI REFERENCE DATA" in prompt


def test_prompt_preserves_legacy_format_when_model_and_requested_tool_match():
    prompt = ModelManager("docker")._build_prompt("show containers")

    assert "Translate to docker command: show containers" in prompt
    assert "requested executable" not in prompt
