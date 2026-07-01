"""Investment-analysis agent built on LangGraph + MCP.

The agent uses a ReAct loop (``create_react_agent``) driven by Claude, with
tools supplied by one or more MCP servers declared in ``mcp_config.json``.

Data source today is the Yahoo Finance MCP server. When CHE MCP is released,
enabling it is a config-only change in ``mcp_config.json`` — this module does
not need to be touched.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "mcp_config.json"

# Default to Opus 4.8 — investment analysis benefits from stronger reasoning.
MODEL = os.environ.get("AGENT_MODEL", "claude-opus-4-8")

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


async def analyze_investments(query: str) -> str:
    """Run one investment-analysis query end to end and return the answer."""
    # Note: `temperature` is deprecated on Opus 4.6+ models (rejected with 400),
    # so it is intentionally not set here.
    llm = ChatAnthropic(model=MODEL)

    client = MultiServerMCPClient(load_mcp_config())
    tools = await client.get_tools()

    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content
