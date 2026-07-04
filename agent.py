"""Investment-analysis agent built on LangGraph + MCP.

The agent runs a ReAct loop implemented as an explicit LangGraph ``StateGraph``.
The **RAG fallback strategy is a first-class node** in the graph: when Claude
invokes the commercial-societies retrieval tool, the router steers that turn to a
dedicated ``rag`` node, distinct from the ``tools`` node that runs the live MCP
data tools::

    START ─▶ agent ──(RAG tool call?)──────▶ rag ──▶ agent ─▶ ... ─▶ END
                 ├──(other tool call?)──────▶ tools ─▶ agent
                 └──(no tool calls)─────────────────────────────▶ END

Nodes wired by hand:

- ``agent``  — calls Claude (with tools bound) on the running message history.
- ``tools``  — executes the live data tools (MCP servers) Claude requested.
- ``rag``    — executes the Spanish commercial-societies retrieval fallback
  (``rag.build_rag_tool()``), which carries its own scope + relevance guardrail.

``_route_after_agent`` loops back to ``agent`` while Claude keeps requesting
tools and routes to ``END`` once it produces a final answer. Splitting the RAG
fallback into its own node (rather than folding it into ``tools``) makes the
retrieval strategy explicit in the graph and gives it a dedicated place to
evolve — building the graph by hand rather than via ``create_react_agent`` is
what leaves that room.

Live-data tools come from one or more MCP servers declared in ``mcp_config.json``
(Yahoo Finance today; CHE MCP is a config-only swap). The RAG fallback is a
local tool added in :func:`_load_tools`; see ``rag.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import rag

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

Fallback knowledge base and scope:
- Your primary job is investment analysis using the live financial tools. As a \
fallback, you also have a document-retrieval tool over a local corpus of \
Argentine commercial-society records (sociedades comerciales — constitution \
deeds, statutes, corporate purpose, directors, capital, Ley 19.550). Those \
source documents are in Spanish.
- Use the retrieval tool only when the financial tools cannot answer and the \
question is about Argentine commercial societies. Ground any such answer in the \
retrieved excerpts and cite the source documents; if the tool reports the query \
is out of scope or returns no relevant records, relay that plainly instead of \
inventing an answer. Respond in the language the user used.
- If a request is neither about markets/investments nor about Argentine \
commercial societies, say it is outside your scope rather than guessing.
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


def resolve_commands(servers: dict[str, Any]) -> dict[str, Any]:
    """Resolve each server's ``command`` to an absolute path on PATH.

    Launching a stdio MCP server spawns its ``command`` as a subprocess. If the
    command isn't on PATH the OS raises a cryptic ``WinError 2`` /
    ``FileNotFoundError``. Resolving up front lets us (a) hand the OS an absolute
    path and (b) fail with an actionable message when the tool is missing.
    """
    for name, cfg in servers.items():
        command = cfg.get("command")
        if not command:
            continue
        resolved = shutil.which(command)
        if resolved is None:
            raise RuntimeError(
                f"MCP server '{name}' requires '{command}', which was not found "
                f"on PATH. Install it (e.g. `winget install --id astral-sh.uv` "
                f"for uvx) and open a NEW terminal so PATH refreshes."
            )
        cfg["command"] = resolved
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


def _route_after_agent(state: AgentState) -> str:
    """Route the agent's turn to the right node.

    - No tool calls          → ``END`` (Claude produced a final answer).
    - Any RAG tool call       → ``rag`` (the commercial-societies fallback).
    - Otherwise               → ``tools`` (live MCP data tools).

    RAG takes precedence when a single turn mixes both (rare, since the domains
    are disjoint); the ``rag`` node still answers every tool call in that turn,
    so no ``tool_use`` is ever left without a ``tool_result``.
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        return END
    if any(call["name"] == rag.TOOL_NAME for call in tool_calls):
        return "rag"
    return "tools"


def build_agent_graph(tools: list, checkpointer=None):
    """Compile the ReAct StateGraph for the given tool set.

    The model and tools are closed over by the node functions, so the returned
    graph is self-contained and can be invoked directly.

    Pass a ``checkpointer`` (e.g. ``MemorySaver``) to persist conversation state
    across invocations; callers must then supply a ``thread_id`` in the invoke
    config so the graph knows which conversation to resume.
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

    async def _run_tool_calls(state: AgentState) -> dict:
        """Execute every tool call in the latest message and append results.

        Shared by the ``tools`` and ``rag`` nodes: the router decides *which*
        node a turn enters, but both resolve calls against the same registry so
        every ``tool_use`` gets a matching ``tool_result`` even in a mixed turn.
        """
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

    async def call_tools(state: AgentState) -> dict:
        """The ``tools`` node: run the live MCP data tools Claude asked for."""
        return await _run_tool_calls(state)

    async def call_rag(state: AgentState) -> dict:
        """The ``rag`` node: run the commercial-societies retrieval fallback.

        A dedicated node so the RAG strategy is explicit in the graph; the tool
        itself enforces the scope + relevance guardrail (see ``rag.py``).
        """
        return await _run_tool_calls(state)

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", call_tools)
    builder.add_node("rag", call_rag)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", _route_after_agent, {"tools": "tools", "rag": "rag", END: END}
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("rag", "agent")
    return builder.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

async def _load_tools() -> list:
    """Return the agent's tools: the MCP server tools plus the RAG fallback.

    MCP remains the primary tool source (live financial data). The local RAG
    tool is appended as a fallback for Argentine commercial-society questions;
    its retrieval index is built/loaded once here, at startup.
    """
    servers = resolve_commands(load_mcp_config())
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    tools.append(rag.build_rag_tool())
    return tools


async def analyze_investments(query: str) -> str:
    """Run one investment-analysis query end to end and return the answer.

    Stateless: each call starts a fresh conversation. For a multi-turn session
    with memory, use :func:`create_session` + :func:`ask` instead.
    """
    tools = await _load_tools()
    graph = build_agent_graph(tools)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return result["messages"][-1].content


async def create_session():
    """Build a reusable, memory-backed agent graph for a multi-turn session.

    Loads the MCP tools once and compiles the graph with an in-memory
    checkpointer. Feed the returned graph to :func:`ask` with a stable
    ``thread_id`` and each turn will see the full prior conversation. Memory
    lives in-process only — it is lost when the process exits.
    """
    tools = await _load_tools()
    return build_agent_graph(tools, checkpointer=MemorySaver())


async def ask(graph, query: str, thread_id: str = "cli-session") -> str:
    """Ask one turn on a memory-backed session graph and return the answer.

    All turns sharing a ``thread_id`` accumulate into one conversation, so
    follow-ups like "compare it to Amazon" resolve against earlier context.
    """
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        },
    )

    return result["messages"][-1].content
