"""Investment-analysis agent built on LangGraph + MCP.

The agent runs a ReAct loop implemented as an explicit LangGraph ``StateGraph``:

    START ─▶ agent ──(tool calls?)──▶ tools ─▶ agent ─▶ ... ─▶ END
                 └────────(no tool calls)───────────────────▶ END

Two nodes are wired by hand:

- ``agent``  — calls Claude (with tools bound) on the running message history.
- ``tools``  — executes whatever tools Claude requested and appends the results.

A conditional edge loops back to ``agent`` while Claude keeps requesting tools,
and routes to ``END`` once it produces a final answer. Building the graph
explicitly (rather than using ``create_react_agent``) leaves room to add nodes
for validation, guardrails, or multi-stage analysis.

Tools come from one or more MCP servers declared in ``mcp_config.json``. Data
source today is the Yahoo Finance MCP server; when CHE MCP is released, enabling
it is a config-only change — this module does not need to be touched.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "mcp_config.json"

# Default to Opus 4.8 — investment analysis benefits from stronger reasoning.
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")

# Safety valve: cap agent/tools cycles so a misbehaving loop can't run forever.
RECURSION_LIMIT = int(os.environ.get("AGENT_RECURSION_LIMIT", "25"))

SYSTEM_PROMPT = """\
You are an investment-analysis assistant. You help the user compare and reason \
about different investment options using live financial data available through \
your tools.

Guidelines:
- Always ground claims in data you retrieve via the tools. If you cannot get a \
number, say so rather than guessing.
- When comparing options, lay out the relevant metrics (price, valuation, \
growth, risk/volatility, analyst views) side by side before concluding.
- Be explicit about time horizon and assumptions.
- State uncertainty and risks plainly. Do not give personalized financial \
advice or guarantees of returns; frame output as analysis, not a recommendation \
to buy or sell.
- Show your reasoning concisely, then give a clear summary the user can act on.
"""

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


# --------------------------------------------------------------------------- #
# MCP config loading
# --------------------------------------------------------------------------- #

def _resolve_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with environment values."""
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    return value


def load_mcp_config() -> dict[str, Any]:
    """Load MCP server config, skipping disabled entries.

    Keys beginning with ``_`` are treated as disabled/placeholder servers
    (e.g. ``_che_mcp_when_released``) and are excluded. ``${VAR}`` placeholders
    in values are resolved from the environment.
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    servers = {
        name: _resolve_env(cfg)
        for name, cfg in raw.items()
        if not name.startswith("_")
    }
    if not servers:
        raise RuntimeError(
            f"No enabled MCP servers in {CONFIG_PATH}. "
            "Enable one by removing its leading underscore."
        )
    return servers


# --------------------------------------------------------------------------- #
# Graph definition
# --------------------------------------------------------------------------- #

class AgentState(TypedDict):
    """Graph state: a growing message history.

    The ``add_messages`` reducer appends new messages returned by each node
    rather than overwriting, which is what makes the ReAct loop accumulate
    context across turns.
    """

    messages: Annotated[list[AnyMessage], add_messages]


def _should_continue(state: AgentState) -> str:
    """Route to the tools node if Claude requested tools, else finish."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_agent_graph(tools: list):
    """Compile the ReAct StateGraph for the given tool set.

    The model and tools are closed over by the node functions, so the returned
    graph is self-contained and can be invoked directly.
    """
    # Note: `temperature` is deprecated on Opus 4.6+ models (rejected with 400),
    # so it is intentionally not set here.
    model = ChatAnthropic(model=MODEL).bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    async def call_model(state: AgentState) -> dict:
        """The ``agent`` node: ask Claude what to do next."""
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await model.ainvoke(messages)
        return {"messages": [response]}

    async def call_tools(state: AgentState) -> dict:
        """The ``tools`` node: run every tool Claude asked for."""
        last = state["messages"][-1]
        observations: list[ToolMessage] = []
        for call in last.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                content = f"Error: unknown tool '{call['name']}'."
            else:
                try:
                    result = await tool.ainvoke(call["args"])
                    content = str(result)
                except Exception as exc:  # keep the loop alive; let Claude adapt
                    content = f"Error running tool '{call['name']}': {exc}"
            observations.append(
                ToolMessage(
                    content=content,
                    name=call["name"],
                    tool_call_id=call["id"],
                )
            )
        return {"messages": observations}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", _should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

async def analyze_investments(query: str) -> str:
    """Run one investment-analysis query end to end and return the answer."""
    client = MultiServerMCPClient(load_mcp_config())
    tools = await client.get_tools()

    graph = build_agent_graph(tools)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return result["messages"][-1].content
