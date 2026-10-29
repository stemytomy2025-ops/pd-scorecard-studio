"""Run the evaluation suite on the downloaded real data.

Usage:
  python scripts/run_eval.py            # tool-level evals (no API key needed)
  python scripts/run_eval.py --agent    # also agent-level evals (calls the Claude API)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from pd_scorecard_studio.config import ROOT, load_config
from pd_scorecard_studio.data import DataError, load_dataset
from pd_scorecard_studio.evaluation import load_cases, run_agent_evals, run_tool_evals
from pd_scorecard_studio.tools import ToolRegistry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="Also run agent-level evals.")
    args = parser.parse_args()

    config = load_config()
    try:
        frames = {ds.id: load_dataset(ds) for ds in config.datasets}
    except DataError as exc:
        print(f"ERROR: {exc}")
        return 1
    registry = ToolRegistry.from_config(config, frames)
    cases = load_cases()
    results = run_tool_evals(cases, registry)

    if args.agent:
        import anthropic

        from pd_scorecard_studio.agent import Agent, describe_data, system_prompt

        agent = Agent(
            anthropic.Anthropic(),
            registry,
            config,
            system_prompt(config, describe_data(config, frames)),
        )
        results += run_agent_evals(cases, agent)

    for r in results:
        print(f"{r.status.upper():<11} {r.level:<6} {r.id:<24} {r.detail}")
    counts = {s: sum(r.status == s for r in results) for s in ("pass", "fail", "not_pinned")}
    print(f"\n{counts['pass']} passed, {counts['fail']} failed, {counts['not_pinned']} not pinned")
    usage = {
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
    }
    if args.agent:
        print(f"Agent token usage: {usage['input_tokens']:,} in / {usage['output_tokens']:,} out")
    report = {
        "run_at": datetime.now(UTC).isoformat(),
        "model": config.model if args.agent else None,
        "usage": usage,
        "results": [r.model_dump() for r in results],
    }
    (ROOT / "evals" / "last_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
