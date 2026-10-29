"""Eval checks, the independent pinning code, and (when downloaded) the real-data evals."""

import importlib.util

import pytest
from conftest import FakeClient, response, text, tool_use

from pd_scorecard_studio.agent import Agent
from pd_scorecard_studio.config import DATA_DIR, ROOT, load_config
from pd_scorecard_studio.data import load_dataset
from pd_scorecard_studio.evaluation import (
    EvalCase,
    Expectation,
    answer_mentions,
    check,
    load_cases,
    run_agent_evals,
    run_tool_evals,
)
from pd_scorecard_studio.tools import ToolRegistry

spec = importlib.util.spec_from_file_location("pin", ROOT / "scripts" / "pin_expectations.py")
pin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pin)

ROWS = [  # SYNTHETIC rows for the pinning code
    {"day": "2024-01-01", "hr": 0, "kind": "a", "n": 10},
    {"day": "2024-01-01", "hr": 1, "kind": "b", "n": 30},
    {"day": "2024-02-02", "hr": 0, "kind": "a", "n": 20},
    {"day": "2024-02-03", "hr": 0, "kind": "b", "n": 40},
]


def test_check_kinds():
    agg = {"top_group": "b", "groups": [{"group": "b", "value": 35.0}]}
    assert check(Expectation(kind="top_group", value="b"), agg)[0] is True
    assert check(Expectation(kind="group_value", group="b", value=35.0), agg)[0] is True
    assert check(Expectation(kind="top_group"), agg)[0] is None  # not pinned
    bt = {"backtest": {"model_mae": 1.0, "baseline_mae": 2.0}}
    assert check(Expectation(kind="model_beats_baseline"), bt)[0] is True
    out = {"found": 1, "rows": [{"robust_z": -4.0}]}
    assert check(Expectation(kind="all_abs_z_at_least", value=3.5), out)[0] is True


def test_pinning_is_computed_independently():
    agg_case = {"args": {"metric": "n", "agg": "mean"}, "expect": {"kind": "top_group"}}
    assert pin.pin_case(agg_case, {"group": "kind"}, ROWS) == "b"
    peak = {"args": {"freq": "month", "agg": "sum"}, "expect": {"kind": "peak_period"}}
    assert pin.pin_case(peak, {"time": "day", "hour": "hr", "metric": "n"}, ROWS) == "2024-02-01"
    count = {"args": {"filters": {"kind": "a"}}, "expect": {"kind": "matched_count"}}
    assert pin.pin_case(count, {}, ROWS) == 2


def test_answer_mentions_numbers_flexibly():
    assert answer_mentions("Peak demand was 1,234.5 rides.", 1234.5)
    assert answer_mentions("Situation 1 is highest.", 1)
    assert not answer_mentions("No number here.", 7)


def test_answer_mentions_dates_in_natural_forms():
    assert answer_mentions("September 2012 had the highest total.", "2012-09-01")
    assert answer_mentions("Peak: Sep 2012.", "2012-09-01")
    assert answer_mentions("The biggest was 28 Oct 2012 at 3am.", "2012-10-28")
    assert answer_mentions("On October 28, 2012 demand spiked.", "2012-10-28")
    assert not answer_mentions("August 2012 had the highest total.", "2012-09-01")
    assert not answer_mentions("September 2011 was busy.", "2012-09-01")


def test_agent_eval_with_fake_client(config, registry):
    tool = next(iter(registry.tools))
    case = EvalCase(
        id="c1", question="q", tool=tool, args={}, expect=Expectation(kind="top_group", value=3)
    )
    client = FakeClient(response(tool_use(tool, {}), stop_reason="tool_use"), response(text("3")))
    agent = Agent(client, registry, config.model_copy(update={"max_turns": 2}), "s")
    [result] = run_agent_evals([case], agent)
    assert result.status in {"pass", "fail"}  # tool may fail on SYNTHETIC data; no crash


def test_cases_reference_configured_tools(config):
    names = {t.name for t in config.tools}
    assert all(case.tool in names for case in load_cases())


def real_data_present() -> bool:
    return all((DATA_DIR / f"{ds.id}.parquet").is_file() for ds in load_config().datasets)


@pytest.mark.skipif(
    not real_data_present(), reason="real data not downloaded (run scripts/fetch_data.py)"
)
def test_tool_evals_pass_on_real_data(config):
    frames = {ds.id: load_dataset(ds) for ds in config.datasets}
    results = run_tool_evals(load_cases(), ToolRegistry.from_config(config, frames))
    failures = [r for r in results if r.status == "fail"]
    assert not failures, failures
