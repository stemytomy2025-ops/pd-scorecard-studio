"""Every configured tool, run on SYNTHETIC data that matches the configured schema."""

import pytest
from conftest import minimal_args

from pd_scorecard_studio.tools import ToolError


def test_every_tool_has_a_valid_definition(registry):
    for definition in registry.definitions():
        assert definition["input_schema"]["type"] == "object"
        assert definition["description"]
        assert definition["input_schema"].get("additionalProperties") is False


def test_every_tool_runs(config, frames, registry):
    for tool in registry.tools.values():
        frame = frames[tool.cfg.dataset_id]
        output = registry.execute(tool.name, minimal_args(tool, frame))
        assert output["operation"] == tool.cfg.operation
        assert output.get("rows_used", 1) > 0 or tool.cfg.operation == "forecast"


def test_unknown_arguments_are_rejected(registry):
    name = next(iter(registry.tools))
    with pytest.raises(ToolError, match="invalid input"):
        registry.execute(name, {"not_a_field": 1})


def test_unknown_tool_is_rejected(registry):
    with pytest.raises(ToolError, match="unknown tool"):
        registry.execute("drop_tables", {})


def test_filters_only_accept_observed_values(registry):
    for tool in registry.tools.values():
        filterable = tool.cfg.params.get("filterable", [])
        if filterable:
            with pytest.raises(ToolError, match="not a value"):
                tool.run({"filters": {filterable[0]: "no-such-value"}})
            with pytest.raises(ToolError, match="cannot filter"):
                tool.run({"filters": {"__nope__": 1}})
            return
    pytest.skip("no tool in this app declares filterable columns")


def test_dates_outside_bounds_and_bad_types_are_rejected(registry, frames):
    for tool in registry.tools.values():
        if tool.cfg.operation == "forecast":
            frame = frames[tool.cfg.dataset_id]
            args = minimal_args(tool, frame) | {"date": "not-a-date"}
            with pytest.raises(ToolError, match="invalid input"):
                tool.run(args)
            args = minimal_args(tool, frame) | {"date": "2999-01-01"}
            with pytest.raises(ToolError, match="between"):
                tool.run(args)
            return
    pytest.skip("no forecast tool in this app")
