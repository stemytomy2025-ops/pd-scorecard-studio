"""Evaluation suite: tool-level checks against pinned expectations, plus agent-level checks.

Expected values in evals/cases.json are pinned by scripts/pin_expectations.py, which recomputes
them from the raw parquet with pyarrow and plain Python, independently of the pandas
operations under test. Property checks (e.g. "model beats baseline") need no pinning.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pd_scorecard_studio.agent import Agent
from pd_scorecard_studio.config import ROOT
from pd_scorecard_studio.tools import ToolError, ToolRegistry

CASES_PATH = ROOT / "evals" / "cases.json"
PINNED_KINDS = {"top_group", "group_value", "peak_period", "matched_count", "overall_bad_rate"}


class Expectation(BaseModel):
    kind: str
    value: Any = None
    group: Any = None
    tolerance: float = 0.005


class EvalCase(BaseModel):
    id: str
    question: str
    tool: str
    args: dict[str, Any] = {}
    expect: Expectation


class EvalResult(BaseModel):
    id: str
    level: str  # "tool" or "agent"
    status: str  # "pass", "fail", "not_pinned", "skipped"
    detail: str
    answer: str = ""  # agent-level only
    input_tokens: int = 0
    output_tokens: int = 0


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(c) for c in data["cases"]]


def _same(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def check(expect: Expectation, output: dict[str, Any]) -> tuple[bool | None, str]:
    """True/False for a verdict, None when the expectation has not been pinned yet."""
    kind, value = expect.kind, expect.value
    if kind in PINNED_KINDS and value is None:
        return None, "expected value not pinned yet (run scripts/pin_expectations.py)"
    if kind == "top_group":
        return _same(output["top_group"], value), f"top_group={output['top_group']!r}"
    if kind == "group_value":
        got = next((g["value"] for g in output["groups"] if _same(g["group"], expect.group)), None)
        ok = got is not None and abs(got - value) <= expect.tolerance * max(1.0, abs(value))
        return ok, f"value for {expect.group!r}={got} (expected {value})"
    if kind == "peak_period":
        got = output["peak"]["period"]
        return str(got).startswith(str(value)), f"peak={got}"
    if kind == "matched_count":
        return output["matched"] == value, f"matched={output['matched']}"
    if kind == "model_beats_baseline":
        bt = output["backtest"]
        return bt["model_mae"] < bt["baseline_mae"], (
            f"model MAE {bt['model_mae']} vs baseline MAE {bt['baseline_mae']}"
        )
    if kind == "all_abs_z_at_least":
        zs = [abs(r["robust_z"]) for r in output["rows"]]
        return all(z >= value for z in zs), f"{output['found']} found, min |z|={min(zs, default=0)}"
    if kind == "overall_bad_rate":
        got = output["overall_bad_rate"]
        return abs(got - value) <= expect.tolerance, f"bad rate={got} (expected {value})"
    if kind == "test_gini_at_least":
        got = output["test"]["gini"]
        return (
            got >= value,
            f"test Gini={got} (floor {value}), train Gini={output['train']['gini']}",
        )
    if kind == "monotonic_bad_rate":
        return output[
            "monotonic_bad_rate"
        ] is True, f"{len(output['bins'])} bins, IV={output['iv']}"
    if kind == "points_sum_to_score":
        total = sum(p["points"] for p in output["points"])
        # Each part is rounded to 4 decimals, so allow that much drift per characteristic.
        ok = abs(total - output["score"]) <= 1e-3 * len(output["points"])
        return ok, f"sum of points={round(total, 4)}, score={output['score']}"
    return False, f"unknown expectation kind {kind!r}"


def run_tool_evals(cases: list[EvalCase], registry: ToolRegistry) -> list[EvalResult]:
    results = []
    for case in cases:
        try:
            output = registry.execute(case.tool, case.args)
        except ToolError as exc:
            results.append(EvalResult(id=case.id, level="tool", status="fail", detail=str(exc)))
            continue
        verdict, detail = check(case.expect, output)
        status = "not_pinned" if verdict is None else "pass" if verdict else "fail"
        results.append(EvalResult(id=case.id, level="tool", status=status, detail=detail))
    return results


def _date_forms(value: Any) -> set[str] | None:
    """Natural ways to write an ISO date: 'September 2012', 'Sep 2012', '28 Oct 2012'..."""
    try:
        d = date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
    month, mon = d.strftime("%B"), d.strftime("%b")
    forms = {
        d.isoformat(),
        f"{d.day} {month} {d.year}",
        f"{d.day} {mon} {d.year}",
        f"{month} {d.day}, {d.year}",
        f"{mon} {d.day}, {d.year}",
    }
    if d.day == 1:  # a month period
        forms |= {f"{month} {d.year}", f"{mon} {d.year}", d.strftime("%Y-%m")}
    return {f.lower() for f in forms}


def answer_mentions(answer: str, value: Any) -> bool:
    if value is None:
        return True
    text = answer.replace(",", "")
    forms = _date_forms(value)
    if forms is not None:
        return any(f.replace(",", "") in text.lower() for f in forms)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).lower() in text.lower()
    candidates = {f"{number:g}", f"{number:.0f}", f"{number:.1f}", f"{number:.2f}"}
    return any(c in text for c in candidates)


def run_agent_evals(cases: list[EvalCase], agent: Agent) -> list[EvalResult]:
    """Pass = the expected tool was called and the pinned value appears in the answer."""
    results = []
    for case in cases:
        run = agent.run(case.question)
        called = [c.name for c in run.tool_calls]
        problems = []
        if run.error:
            problems.append(f"agent error: {run.error}")
        if case.tool not in called:
            problems.append(f"expected tool {case.tool} but called {called}")
        shown = case.expect.value if case.expect.kind in PINNED_KINDS else None
        if shown is not None and not answer_mentions(run.answer, shown):
            problems.append(f"answer does not mention {shown!r}")
        status = "fail" if problems else "pass"
        detail = "; ".join(problems) or f"called {called}"
        results.append(
            EvalResult(
                id=case.id,
                level="agent",
                status=status,
                detail=detail,
                answer=run.answer,
                input_tokens=run.input_tokens,
                output_tokens=run.output_tokens,
            )
        )
    return results
