"""A bounded tool-using agent on the Anthropic Messages API.

Manual loop (rather than the beta tool runner) so the app controls the budget and can stream
progress events to the UI: at most `max_turns` model calls and `max_tool_calls` tool calls.
Tool failures go back to the model as `is_error` results so it can correct its arguments.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pd_scorecard_studio.config import AppConfig
from pd_scorecard_studio.tools import ToolError, ToolRegistry

MAX_RESULT_CHARS = 12_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    seconds: float = 0.0


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


Event = Callable[[str, dict[str, Any]], None]


def describe_data(config: AppConfig, frames: dict[str, pd.DataFrame]) -> str:
    """Facts about the loaded data for the system prompt (computed, not assumed)."""
    lines = []
    for ds in config.datasets:
        df = frames[ds.id]
        dates = [c for c, kind in ds.expected_schema.items() if kind in ("date", "datetime")]
        span = ""
        if dates:
            ts = pd.to_datetime(df[dates[0]])
            span = f", covering {ts.min().date()} to {ts.max().date()}"
        lines.append(
            f"- {ds.name}: {len(df):,} rows{span}. Licence {ds.license}. "
            f"Attribution: {ds.attribution}"
        )
        lines += [f"  - Caveat: {item}" for item in ds.limitations]
    return "\n".join(lines)


def system_prompt(config: AppConfig, data_description: str) -> str:
    return f"""You are the analytics assistant inside "{config.title}". You answer questions
about this data only:
{data_description}

Rules:
- Every number you state must come from a tool result in this conversation. If no tool can
  answer, say so and suggest what data would be needed. Never estimate from memory.
- The data is historical. Interpret "tomorrow" or "next week" relative to the last date in
  the data and say that you did so.
- Prefer one or two well-chosen tool calls. You may call at most {config.max_tool_calls} tools.
- If a tool returns an error, read it, fix the arguments, and try once more.
- Answer in a few sentences: the result first, then the key numbers, then caveats (for
  forecasts, mention the backtest error against the baseline)."""


class Agent:
    def __init__(self, client: Any, registry: ToolRegistry, config: AppConfig, system: str):
        self.client = client
        self.registry = registry
        self.config = config
        self.system = system

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": 16000,
            "system": self.system,
            "tools": self.registry.definitions(),
            "messages": messages,
            "output_config": {"effort": self.config.effort},
        }
        if self.config.use_refusal_fallback:
            # Re-runs a declined request on a fallback model inside the same call.
            return self.client.beta.messages.create(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            )
        return self.client.messages.create(**params)

    def run(self, question: str, on_event: Event | None = None) -> AgentResult:
        emit = on_event or (lambda kind, data: None)
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        result = AgentResult(answer="")
        for turn in range(1, self.config.max_turns + 1):
            emit("thinking", {"turn": turn})
            response = self._create(messages)
            result.turns = turn
            result.stop_reason = response.stop_reason
            usage = getattr(response, "usage", None)
            result.input_tokens += getattr(usage, "input_tokens", 0) or 0
            result.output_tokens += getattr(usage, "output_tokens", 0) or 0
            text = "\n".join(b.text for b in response.content if b.type == "text").strip()

            if response.stop_reason == "refusal":
                result.answer = text or "The model declined to answer this request."
                result.error = "refusal"
                return result
            if response.stop_reason == "max_tokens":
                result.answer = text
                result.error = "the response hit the max_tokens limit"
                return result

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                result.answer = text
                return result

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_uses:
                if len(result.tool_calls) >= self.config.max_tool_calls:
                    call = ToolCall(block.name, dict(block.input), error="tool-call budget used")
                else:
                    call = self._execute(block.name, dict(block.input), emit)
                result.tool_calls.append(call)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": call.error
                        or json.dumps(call.output, default=str)[:MAX_RESULT_CHARS],
                        "is_error": call.error is not None,
                    }
                )
            # All results for one assistant turn go back in a single user message.
            messages.append({"role": "user", "content": tool_results})

        result.error = f"stopped after {self.config.max_turns} model turns without a final answer"
        result.answer = result.answer or "I could not finish within the allowed number of steps."
        return result

    def _execute(self, name: str, tool_input: dict[str, Any], emit: Event) -> ToolCall:
        call = ToolCall(name=name, input=tool_input)
        emit("tool_start", {"name": name, "input": tool_input})
        started = time.perf_counter()
        try:
            call.output = self.registry.execute(name, tool_input)
        except ToolError as exc:
            call.error = str(exc)
        call.seconds = round(time.perf_counter() - started, 3)
        emit("tool_end", {"name": name, "error": call.error, "seconds": call.seconds})
        return call
