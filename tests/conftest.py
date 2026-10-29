"""Test fixtures. All data here is SYNTHETIC, generated from the configured schema.

Tests never touch the network or the Anthropic API. Real-data checks live in
test_evaluation.py and are skipped when data/raw has not been downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pd_scorecard_studio.config import AppConfig, DatasetConfig, load_config
from pd_scorecard_studio.tools import ToolRegistry

HOUR_NAMES = {"hr", "hour"}


def synthetic_frame(ds: DatasetConfig, binary: set[str] = frozenset()) -> pd.DataFrame:
    """A deterministic SYNTHETIC frame that satisfies `ds.expected_schema`.

    Columns in `binary` (e.g. a scorecard's default flag) get 0/1 values with both classes.
    """
    schema = ds.expected_schema
    hourly = any(c.lower() in HOUR_NAMES and t == "integer" for c, t in schema.items())
    n = 24 * 60 if hourly else 200
    i = np.arange(n)
    day = i // 24 if hourly else i
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for column, kind in schema.items():
        if kind == "date":
            data[column] = pd.Timestamp("2024-01-01") + pd.to_timedelta(day, unit="D")
        elif kind == "datetime":
            step = "h" if hourly else "D"
            data[column] = pd.date_range("2024-01-01", periods=n, freq=step)
        elif column in binary:
            data[column] = (rng.random(n) < 0.3).astype(int)
        elif kind == "integer" and column.lower() in HOUR_NAMES:
            data[column] = i % 24
        elif kind == "integer":
            data[column] = 10 + (i % 24) * 3 + (day % 7) + rng.integers(0, 3, n)
        elif kind == "float":
            data[column] = np.round(0.2 + (i % 10) / 20, 3)
        elif kind == "boolean":
            data[column] = i % 2 == 0
        else:
            data[column] = np.array(["A", "B", "C"])[i % 3]
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def config() -> AppConfig:
    """The real app config, with row minimums relaxed: SYNTHETIC frames are small."""
    real = load_config()
    datasets = [ds.model_copy(update={"expected_min_rows": 1}) for ds in real.datasets]
    return real.model_copy(update={"datasets": datasets})


@pytest.fixture(scope="session")
def frames(config: AppConfig) -> dict[str, pd.DataFrame]:
    targets = {(t.dataset_id, t.params["target"]) for t in config.tools if "target" in t.params}
    return {
        ds.id: synthetic_frame(ds, {col for ds_id, col in targets if ds_id == ds.id})
        for ds in config.datasets
    }


@pytest.fixture(scope="session")
def registry(config: AppConfig, frames) -> ToolRegistry:
    return ToolRegistry.from_config(config, frames)


def minimal_args(tool, frame: pd.DataFrame) -> dict:
    """Smallest valid input for a tool (forecast needs a date, and an hour if hourly)."""
    if tool.cfg.operation != "forecast":
        return {}
    last = pd.to_datetime(frame[tool.cfg.params["time"]]).max().date()
    args = {"date": last.isoformat()}
    if tool.cfg.params.get("hour"):
        args["hour"] = 8
    return args


# ---- fake Anthropic client --------------------------------------------------------


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(name: str, tool_input: dict, block_id: str = "tu_1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def response(*blocks, stop_reason: str = "end_turn") -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=100, output_tokens=20)
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=usage)


class FakeMessages:
    def __init__(self, scripted: list):
        self.scripted = list(scripted)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # Snapshot messages: the agent appends to the same list after this call.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.scripted.pop(0)


class FakeClient:
    """Stands in for anthropic.Anthropic(); scripted responses, records every request."""

    def __init__(self, *scripted):
        self.messages = FakeMessages(scripted)
        self.beta = SimpleNamespace(messages=self.messages)
