"""The agent loop against a scripted fake client (no network, no API key)."""

from conftest import FakeClient, minimal_args, response, text, tool_use

from pd_scorecard_studio.agent import FALLBACK_BETA, Agent


def first_tool(registry, frames):
    tool = next(iter(registry.tools.values()))
    return tool, minimal_args(tool, frames[tool.cfg.dataset_id])


def make_agent(config, registry, client, **overrides):
    return Agent(client, registry, config.model_copy(update=overrides), "system prompt")


def test_tool_call_then_answer(config, registry, frames):
    tool, args = first_tool(registry, frames)
    client = FakeClient(
        response(tool_use(tool.name, args), stop_reason="tool_use"),
        response(text("The answer is 42.")),
    )
    result = make_agent(config, registry, client).run("question?")
    assert result.ok and result.answer == "The answer is 42."
    assert [c.name for c in result.tool_calls] == [tool.name]
    assert result.tool_calls[0].output["operation"] == tool.cfg.operation
    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tu_1" and tool_result["is_error"] is False
    assert result.turns == 2 and result.input_tokens == 200


def test_invalid_tool_input_is_returned_as_error(config, registry, frames):
    tool, _ = first_tool(registry, frames)
    client = FakeClient(
        response(tool_use(tool.name, {"bogus": True}), stop_reason="tool_use"),
        response(text("Sorry, I could not compute that.")),
    )
    result = make_agent(config, registry, client).run("question?")
    assert result.tool_calls[0].error and "invalid input" in result.tool_calls[0].error
    tool_result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True


def test_unknown_tool_is_an_error_not_a_crash(config, registry):
    client = FakeClient(
        response(tool_use("delete_everything", {}), stop_reason="tool_use"),
        response(text("done")),
    )
    result = make_agent(config, registry, client).run("q")
    assert "unknown tool" in result.tool_calls[0].error


def test_turn_limit_stops_the_loop(config, registry, frames):
    tool, args = first_tool(registry, frames)
    looping = [response(tool_use(tool.name, args), stop_reason="tool_use") for _ in range(3)]
    result = make_agent(config, registry, FakeClient(*looping), max_turns=3).run("q")
    assert not result.ok and "3 model turns" in result.error


def test_tool_call_budget(config, registry, frames):
    tool, args = first_tool(registry, frames)
    client = FakeClient(
        response(
            tool_use(tool.name, args, "a"), tool_use(tool.name, args, "b"), stop_reason="tool_use"
        ),
        response(text("ok")),
    )
    result = make_agent(config, registry, client, max_tool_calls=1).run("q")
    assert result.tool_calls[0].error is None
    assert result.tool_calls[1].error == "tool-call budget used"
    # Both results go back in one user message.
    assert len(client.messages.calls[1]["messages"][-1]["content"]) == 2


def test_refusal_is_reported(config, registry):
    client = FakeClient(response(text(""), stop_reason="refusal"))
    result = make_agent(config, registry, client).run("q")
    assert result.error == "refusal"


def test_fallback_setting_controls_request_shape(config, registry):
    with_fallback = FakeClient(response(text("hi")))
    make_agent(config, registry, with_fallback, use_refusal_fallback=True).run("q")
    call = with_fallback.messages.calls[0]
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["model"] == config.model and call["output_config"] == {"effort": config.effort}

    plain = FakeClient(response(text("hi")))
    make_agent(config, registry, plain, use_refusal_fallback=False).run("q")
    assert "betas" not in plain.messages.calls[0]
