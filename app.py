"""Streamlit UI: ask the agent, run tools directly, run evals, and inspect data provenance."""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
import pandas as pd
import streamlit as st

from pd_scorecard_studio.agent import Agent, AgentResult, describe_data, system_prompt
from pd_scorecard_studio.config import load_config
from pd_scorecard_studio.data import DataError, fetch_dataset, load_dataset, load_manifest
from pd_scorecard_studio.evaluation import load_cases, run_tool_evals
from pd_scorecard_studio.tools import ToolError, ToolRegistry

CONFIG = load_config()
st.set_page_config(page_title=CONFIG.title, page_icon="📊", layout="wide")


def api_key() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets.toml configured
        return None


def data_stamp() -> str:
    manifests = [load_manifest(ds) for ds in CONFIG.datasets]
    return "|".join(m["fetched_at"] if m else "missing" for m in manifests)


@st.cache_resource(show_spinner="Loading and validating data")
def runtime(stamp: str) -> tuple[dict[str, pd.DataFrame], ToolRegistry]:
    frames = {ds.id: load_dataset(ds) for ds in CONFIG.datasets}
    return frames, ToolRegistry.from_config(CONFIG, frames)


def render_output(output: dict[str, Any]) -> None:
    op = output.get("operation")
    if op == "trend":
        chart = pd.DataFrame(output["points"]).set_index("period")
        st.line_chart(chart, y="value")
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak", output["peak"]["value"], help=output["peak"]["period"])
        c2.metric("Trough", output["trough"]["value"], help=output["trough"]["period"])
        c3.metric("Periods", output["periods"])
    elif op == "aggregate":
        chart = pd.DataFrame(output["groups"]).astype({"group": str}).set_index("group")
        st.bar_chart(chart, y="value")
    elif op == "forecast":
        bt = output["backtest"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast", output["prediction"])
        c2.metric("Baseline", output["baseline_prediction"])
        c3.metric("Backtest MAE (model / baseline)", f"{bt['model_mae']} / {bt['baseline_mae']}")
        if output["beyond_data"]:
            st.caption("This date is after the end of the data: it is an extrapolation.")
    elif op == "woe_iv" and "bins" in output:
        c1, c2 = st.columns(2)
        c1.metric("Information Value", output["iv"], help=output["strength"])
        c2.metric("Monotonic bad rate", "yes" if output["monotonic_bad_rate"] else "no")
        table = pd.DataFrame(output["bins"]).astype({"bin": str})
        st.bar_chart(table.set_index("bin"), y="woe")
        st.dataframe(table, use_container_width=True, hide_index=True)
    elif op == "woe_iv":
        table = pd.DataFrame(output["ranking"])
        st.bar_chart(table.set_index("feature"), y="iv")
        st.dataframe(table, use_container_width=True, hide_index=True)
    elif op == "scorecard_performance":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Gini", output["test"]["gini"], help=f"Train {output['train']['gini']}")
        c2.metric("Test KS", output["test"]["ks"])
        c3.metric("Test AUC", output["test"]["auc"])
        c4.metric("Max calibration gap", output["max_calibration_gap"])
        calibration = pd.DataFrame(output["calibration_test"]).set_index("band")
        st.line_chart(calibration, y=["predicted_pd", "observed_bad_rate"])
        st.dataframe(calibration, use_container_width=True)
    elif op == "score_breakdown":
        c1, c2 = st.columns(2)
        c1.metric("Score", output["score"])
        c2.metric("PD", output["pd"])
        table = pd.DataFrame(output["points"])
        st.bar_chart(table.set_index("feature"), y="points")
        st.dataframe(table, use_container_width=True, hide_index=True)
    if output.get("rows"):
        st.dataframe(pd.DataFrame(output["rows"]), use_container_width=True, hide_index=True)
    with st.expander("Raw tool output"):
        st.json(output)


def render_result(result: AgentResult) -> None:
    st.markdown(result.answer or "_No answer text returned._")
    if result.error:
        st.warning(f"The run ended early: {result.error}")
    label = f"How this answer was computed ({len(result.tool_calls)} tool calls)"
    with st.expander(label):
        for call in result.tool_calls:
            st.markdown(f"**`{call.name}`** · {call.seconds}s")
            st.code(json.dumps(call.input, indent=2, default=str), language="json")
            if call.error:
                st.error(call.error)
            elif call.output:
                render_output(call.output)
        st.caption(
            f"{result.turns} model turns · {result.input_tokens:,} input / "
            f"{result.output_tokens:,} output tokens · stop reason: {result.stop_reason}"
        )


def ask(question: str, frames: dict[str, pd.DataFrame], registry: ToolRegistry) -> None:
    key = api_key()
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    agent = Agent(client, registry, CONFIG, system_prompt(CONFIG, describe_data(CONFIG, frames)))
    with st.status("Working on it", expanded=True) as status:

        def on_event(kind: str, data: dict[str, Any]) -> None:
            if kind == "thinking":
                status.update(label=f"Thinking (step {data['turn']})")
            elif kind == "tool_start":
                status.write(f"Running `{data['name']}` with `{json.dumps(data['input'])}`")
            elif kind == "tool_end":
                mark = "✗ " + data["error"] if data["error"] else f"✓ done in {data['seconds']}s"
                status.write(mark)

        error: str | None = None
        try:
            result = agent.run(question, on_event)
        except anthropic.AuthenticationError:
            error = "The Anthropic API rejected the credentials. Check ANTHROPIC_API_KEY."
        except anthropic.RateLimitError:
            error = "The Anthropic API is rate limiting requests. Wait a moment and retry."
        except anthropic.APIStatusError as exc:
            error = f"The Anthropic API returned HTTP {exc.status_code}: {exc.message}"
        except anthropic.APIConnectionError:
            error = "Could not reach the Anthropic API. Check the network connection."
        except Exception as exc:  # e.g. no credentials configured at all
            error = f"{type(exc).__name__}: {exc}"
        if error:
            status.update(label="Could not run the assistant", state="error", expanded=False)
        else:
            status.update(
                label="Done" if result.ok else "Finished with a warning",
                state="complete" if result.ok else "error",
                expanded=False,
            )
    # Shown outside the (collapsed) status box so the reason is always visible.
    if error:
        st.error(error)
        if not api_key():
            st.info(
                "Set ANTHROPIC_API_KEY in your environment or .streamlit/secrets.toml, then "
                "retry. The Run a tool tab works without a key."
            )
        return
    render_result(result)
    st.session_state.history.append((question, result))


# ---------------------------------------------------------------- sidebar

# ---------------------------------------------------------------- main
st.title(CONFIG.title)
st.caption(CONFIG.tagline)

try:
    frames, registry = runtime(data_stamp())
except DataError as exc:
    st.warning(str(exc))
    st.info("Use **Download / refresh data** in the sidebar to fetch the documented dataset.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

ask_tab, data_tab = st.tabs(
    ["Ask the assistant", "Data & sources"]
)

with ask_tab:
    st.markdown("Try an example, or ask your own question about the data:")
    for i, example in enumerate(CONFIG.example_questions):
        # One per row so long questions are never truncated.
        if st.button(example, key=f"example-{i}", use_container_width=True):
            st.session_state.pending = example
    for past_question, past_result in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(past_question)
        with st.chat_message("assistant"):
            render_result(past_result)
    question = st.chat_input("Ask about the data") or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            ask(question, frames, registry)

# with tools_tab:
#     st.markdown("Run any tool directly with validated arguments (no API key needed).")
#     name = st.selectbox("Tool", list(registry.tools))
#     tool = registry.tools[name]
#     st.caption(tool.cfg.description)
#     with st.expander("Input schema"):
#         st.json(tool.input_model.model_json_schema())
#     examples = {c.tool: c.args for c in load_cases()}
#     raw = st.text_area(
#         "Arguments (JSON)", json.dumps(examples.get(name, {}), indent=2), key=f"args-{name}"
#     )
#     if st.button("Run tool", type="primary"):
#         try:
#             output = registry.execute(name, json.loads(raw or "{}"))
#         except json.JSONDecodeError as exc:
#             st.error(f"Arguments are not valid JSON: {exc}")
#         except ToolError as exc:
#             st.error(str(exc))
#         else:
#             render_output(output)

# with eval_tab:
#     st.markdown(
#         "Tool-level evals compare each tool's output on the real data with expected values "
#         "pinned by an independent computation (`scripts/pin_expectations.py`). Agent-level "
#         "evals need an API key: `python scripts/run_eval.py --agent`."
#     )
#     if st.button("Run tool-level evals"):
#         with st.spinner("Running evals"):
#             results = run_tool_evals(load_cases(), registry)
#         table = pd.DataFrame([r.model_dump() for r in results])
#         st.dataframe(table, use_container_width=True, hide_index=True)
#         passed = sum(r.status == "pass" for r in results)
#         st.metric("Passed", f"{passed} / {len(results)}")

with data_tab:
    for ds in CONFIG.datasets:
        st.subheader(ds.name)
        st.markdown(f"**Licence:** {ds.license}  \n**Attribution:** {ds.attribution}")
        st.markdown(f"**Source:** {ds.landing_page}")
        # if ds.limitations:
        #     st.markdown("**Known limitations**")
        #     for item in ds.limitations:
        #         st.markdown(f"- {item}")
        schema = pd.DataFrame(
            [{"column": c, "expected type": t} for c, t in ds.expected_schema.items()]
        )
        st.markdown("**Validated schema**")
        st.dataframe(schema, hide_index=True)
        manifest = load_manifest(ds)
        if manifest:
            st.markdown("**Download manifest**")
            st.json(manifest)
